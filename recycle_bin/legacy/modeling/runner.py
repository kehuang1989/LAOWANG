# -*- coding: utf-8 -*-
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import logging
from typing import List, Optional, Sequence, Tuple

from sqlalchemy import text
from tqdm import tqdm

from . import db as mdb
from .base import Model


def ensure_tables(engine, models: Sequence[Model]) -> None:  # noqa: ANN001
    mdb.ensure_model_runs_table(engine)
    for m in models:
        m.ensure_tables(engine)


def _latest_stock_daily_date(engine) -> Optional[str]:  # noqa: ANN001
    return mdb.latest_stock_daily_date(engine)


def _delete_model_runs(engine, *, model_name: str) -> None:  # noqa: ANN001
    with engine.begin() as conn:
        conn.execute(
            text("DELETE FROM model_runs WHERE model_name = :m"),
            {"m": model_name},
        )


def _normalize_date_arg(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    try:
        return dt.datetime.strptime(s, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {value}. Expect YYYYMMDD or YYYY-MM-DD") from exc


def _normalize_range(
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    start_norm = _normalize_date_arg(start_date)
    end_norm = _normalize_date_arg(end_date)
    if start_norm and end_norm and start_norm > end_norm:
        raise ValueError(f"start-date {start_norm} is after end-date {end_norm}")
    return start_norm, end_norm


def _filter_dates(
    dates: Sequence[str],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[str]:
    out: List[str] = []
    for d in dates:
        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        out.append(str(d))
    return out


def _worker_budget(workers: int, *, outer_cap: int) -> Tuple[int, int]:
    """
    We have two nested concurrency layers:
    - outer: across trade dates
    - inner: passed down into each model's compute(), which may use threads

    Keep the product bounded to avoid thread explosions in "full" runs.
    """
    total = max(1, int(workers))
    outer = max(1, min(int(outer_cap), total))
    inner = max(1, total // outer)
    inner = min(64, inner)
    return int(outer), int(inner)


def _run_dates(
    *,
    engine,
    models: Sequence[Model],
    dates: Sequence[str],
    workers: int,
    outer_cap: int,
    skip_if_ok: bool,
    desc: str,
) -> None:
    if not dates:
        logging.info("%s: no trading days to process.", desc)
        return

    date_workers, model_workers = _worker_budget(int(workers), outer_cap=outer_cap)
    logging.info(
        "%s: dates=%d outer_workers=%d model_workers=%d range=%s..%s",
        desc,
        len(dates),
        date_workers,
        model_workers,
        dates[0],
        dates[-1],
    )

    progress = tqdm(total=len(dates), desc=desc, leave=False)

    def run_date(d: str) -> None:
        for m in models:
            if skip_if_ok and mdb.is_model_ok(engine, model_name=m.name, trade_date=d):
                continue

            started = mdb.now_ts()
            try:
                logging.info("[%s] compute %s", m.name, d)
                df = m.compute(engine=engine, trade_date=d, workers=int(model_workers))
                n = m.save(engine=engine, trade_date=d, df=df)
                mdb.write_model_run(
                    engine,
                    model_name=m.name,
                    trade_date=d,
                    status="ok",
                    row_count=int(n),
                    message="",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
                logging.info("[%s] ok %s rows=%d", m.name, d, n)
            except Exception as e:  # noqa: BLE001
                mdb.write_model_run(
                    engine,
                    model_name=m.name,
                    trade_date=d,
                    status="error",
                    row_count=0,
                    message=f"{type(e).__name__}: {e}",
                    started_at=started,
                    finished_at=mdb.now_ts(),
                )
                logging.exception("[%s] failed %s", m.name, d)

    if date_workers <= 1 or len(dates) <= 1:
        for d in dates:
            try:
                run_date(str(d))
            finally:
                progress.update(1)
        progress.close()
        return

    with ThreadPoolExecutor(max_workers=int(date_workers)) as ex:
        futs = {ex.submit(run_date, str(d)): str(d) for d in dates}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception:  # noqa: BLE001
                logging.exception("Model worker crashed for date=%s", futs.get(f))
            finally:
                progress.update(1)
    progress.close()


def update_models(
    *,
    engine,
    models: Sequence[Model],
    workers: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """
    Incremental update:
    - Determine latest trading day from stock_daily.
    - For each model, compute dates after its last_ok_date; if never ran,
      compute latest date only.
    """
    ensure_tables(engine, models)

    latest = _latest_stock_daily_date(engine)
    if not latest:
        logging.warning("No stock_daily data; skip models update.")
        return

    start_norm, end_norm = _normalize_range(start_date, end_date)

    date_set: set[str] = set()
    for m in models:
        last_ok = mdb.last_ok_date(engine, model_name=m.name)
        if last_ok:
            target_end = end_norm or latest
            dates = mdb.list_stock_daily_dates(engine, after_date=last_ok, end_date=target_end)
        else:
            target_latest = end_norm or latest
            if target_latest:
                dates = [target_latest]
            else:
                dates = []
        for d in dates:
            date_set.add(str(d))

    dates_all = _filter_dates(sorted(date_set), start_norm, end_norm)
    if not dates_all:
        logging.info("No new trading days for models.")
        return

    _run_dates(
        engine=engine,
        models=models,
        dates=dates_all,
        workers=workers,
        outer_cap=4,
        skip_if_ok=True,
        desc="models incremental",
    )


def full_recompute(
    *,
    engine,
    models: Sequence[Model],
    workers: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> None:
    """
    Full recompute for trading days in stock_daily.
    When start_date/end_date are provided, the run is constrained to that range.
    Warning: can be very slow if a model is heavy for historical backfill.
    """
    ensure_tables(engine, models)
    latest = _latest_stock_daily_date(engine)
    if not latest:
        logging.warning("No stock_daily data; skip models full recompute.")
        return

    start_norm, end_norm = _normalize_range(start_date, end_date)
    effective_end = end_norm or latest

    dates = mdb.list_stock_daily_dates(engine, start_date=start_norm, end_date=effective_end)
    if not dates:
        logging.warning("No stock_daily dates found for the requested range.")
        return

    if not start_norm and not end_norm:
        for m in models:
            _delete_model_runs(engine, model_name=m.name)
    else:
        logging.info(
            "Full recompute constrained to %s..%s (existing rows will be overwritten per date).",
            start_norm or "-",
            end_norm or "-",
        )

    _run_dates(
        engine=engine,
        models=models,
        dates=dates,
        workers=workers,
        outer_cap=32,
        skip_if_ok=False,
        desc="models full",
    )
