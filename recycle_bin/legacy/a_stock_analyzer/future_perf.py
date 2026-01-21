# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from . import db


@dataclass(frozen=True)
class FuturePerfConfig:
    ne_list: list[int]
    min_score: float = 80.0
    workers: int = 8
    top_n: Optional[int] = None


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_trading_days(engine) -> list[str]:
    with engine.connect() as conn:
        days = [r[0] for r in conn.execute(text("SELECT DISTINCT date FROM stock_daily ORDER BY date"))]
    return [str(d) for d in days if d]


def _pick_signal_date(trading_days: list[str], ne: int) -> str:
    if len(trading_days) <= ne:
        raise RuntimeError(f"Not enough trading days: days={len(trading_days)} <= ne={ne}")
    return trading_days[-ne - 1]


def _load_signals(engine, signal_date: str, *, min_score: float, top_n: Optional[int]) -> list[tuple[str, str, float]]:
    sql = """
        SELECT s.stock_code, COALESCE(i.name, ''), s.total_score
        FROM stock_scores_v3 s
        LEFT JOIN stock_info i ON i.stock_code = s.stock_code
        WHERE s.score_date = :d
          AND s.total_score >= :k
          AND (s.status_tags IS NULL OR s.status_tags NOT LIKE '%RISK_FILTERED%')
        ORDER BY s.total_score DESC
    """
    if top_n is not None and int(top_n) > 0:
        sql += " LIMIT :top_n"
        params = {"d": signal_date, "k": float(min_score), "top_n": int(top_n)}
    else:
        params = {"d": signal_date, "k": float(min_score)}

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [(str(code), str(name), float(score)) for code, name, score in rows if code]


def _eval_one(engine, stock_code: str, name: str, score: float, signal_date: str, ne: int) -> Optional[dict]:
    with engine.connect() as conn:
        row0 = conn.execute(
            text("SELECT close FROM stock_daily WHERE stock_code = :c AND date = :d"),
            {"c": stock_code, "d": signal_date},
        ).fetchone()
        if not row0 or row0[0] is None:
            return None

        start_close = float(row0[0])
        if not (start_close > 0):
            return None

        rows = conn.execute(
            text(
                """
                SELECT date, high, low, close
                FROM stock_daily
                WHERE stock_code = :c AND date > :d
                ORDER BY date ASC
                LIMIT :ne
                """
            ),
            {"c": stock_code, "d": signal_date, "ne": int(ne)},
        ).fetchall()

    if len(rows) < ne:
        return None

    highs = []
    lows = []
    closes = []
    for _, high, low, close in rows:
        c = float(close) if close is not None else None
        if c is None:
            return None
        closes.append(c)
        highs.append(float(high) if high is not None else c)
        lows.append(float(low) if low is not None else c)

    max_price = max(highs)
    min_price = min(lows)
    final_price = closes[-1]
    end_date = str(rows[-1][0])

    return {
        "stock_code": stock_code,
        "name": name,
        "score": float(score),
        "signal_date": signal_date,
        "ne": int(ne),
        "start_close": start_close,
        "end_date": end_date,
        "max_price": max_price,
        "min_price": min_price,
        "final_price": final_price,
        "max_return": max_price / start_close - 1.0,
        "min_return": min_price / start_close - 1.0,
        "final_return": final_price / start_close - 1.0,
    }


def _write_report(md_path: Path, cfg: FuturePerfConfig, df: pd.DataFrame) -> None:
    db.ensure_parent_dir(md_path)

    lines: list[str] = []
    lines.append("# Signal Future Performance Report")
    lines.append("")
    lines.append(f"- Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- ne_list: {cfg.ne_list}")
    lines.append(f"- min_score(k): {cfg.min_score}")
    lines.append(f"- workers: {cfg.workers}")
    lines.append(f"- top_n: {cfg.top_n}")
    lines.append("")

    if df is None or df.empty:
        lines.append("_No rows matched._")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    summary = (
        df.groupby(["signal_date", "ne"])
        .agg(
            n=("stock_code", "count"),
            max_return_mean=("max_return", "mean"),
            min_return_mean=("min_return", "mean"),
            final_return_mean=("final_return", "mean"),
            max_return_median=("max_return", "median"),
            min_return_median=("min_return", "median"),
            final_return_median=("final_return", "median"),
        )
        .reset_index()
        .sort_values(["signal_date", "ne"])
    )

    def to_md_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        out = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in frame.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)

    lines.append("## Summary")
    lines.append("")
    lines.append(to_md_table(summary))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Returns are computed as (price / start_close - 1).")
    lines.append("- Window is the next `ne` rows in `stock_daily` after `signal_date` (stock-specific trading days).")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_future_perf(
    db_target: str,
    *,
    ne_list: Iterable[int] = (5, 10, 20, 30),
    min_score: float = 80.0,
    signal_date: Optional[str] = None,
    workers: int = 8,
    top_n: Optional[int] = None,
    out_dir: Path = Path("output"),
) -> tuple[Path, Path]:
    cfg = FuturePerfConfig(
        ne_list=[int(x) for x in ne_list],
        min_score=float(min_score),
        workers=max(1, int(workers)),
        top_n=int(top_n) if top_n is not None else None,
    )

    engine = db.make_engine(
        db_target,
        pool_size=max(5, cfg.workers * 2),
        max_overflow=cfg.workers * 2,
    )
    db.init_db(engine)

    trading_days = _load_trading_days(engine)
    if not trading_days:
        raise RuntimeError("No trading days found in stock_daily; run pipeline first.")

    jobs: list[tuple[str, int]] = []
    if signal_date:
        for ne in cfg.ne_list:
            jobs.append((signal_date, int(ne)))
    else:
        for ne in cfg.ne_list:
            jobs.append((_pick_signal_date(trading_days, int(ne)), int(ne)))

    rows_all: list[dict] = []
    for d, ne in jobs:
        signals = _load_signals(engine, d, min_score=cfg.min_score, top_n=cfg.top_n)
        if not signals:
            logging.info("No signals for %s (ne=%d, k=%.1f).", d, ne, cfg.min_score)
            continue

        eval_jobs = [(code, name, score) for code, name, score in signals]

        if cfg.workers <= 1:
            for code, name, score in tqdm(eval_jobs, desc=f"Eval {d} ne={ne}", unit="stk"):
                r = _eval_one(engine, code, name, score, d, ne)
                if r is not None:
                    rows_all.append(r)
        else:
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                futs = [
                    ex.submit(_eval_one, engine, code, name, score, d, ne) for code, name, score in eval_jobs
                ]
                for f in tqdm(as_completed(futs), total=len(futs), desc=f"Eval {d} ne={ne}", unit="stk"):
                    try:
                        r = f.result()
                        if r is not None:
                            rows_all.append(r)
                    except Exception:  # noqa: BLE001
                        logging.exception("Future perf worker failed (date=%s, ne=%d)", d, ne)

    df = pd.DataFrame(rows_all)
    ts = _timestamp()
    csv_path = out_dir / f"future_perf_{ts}.csv"
    md_path = out_dir / f"future_perf_{ts}.md"
    db.ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    _write_report(md_path, cfg, df)

    # Persist into stock_future_perf
    if df is not None and not df.empty:
        with engine.begin() as conn:
            for r in rows_all:
                db.upsert_future_perf(
                    conn,
                    stock_code=r["stock_code"],
                    signal_date=r["signal_date"],
                    ne=int(r["ne"]),
                    max_price=float(r["max_price"]),
                    min_price=float(r["min_price"]),
                    final_price=float(r["final_price"]),
                    max_return=float(r["max_return"]),
                    min_return=float(r["min_return"]),
                    final_return=float(r["final_return"]),
                )

    logging.info("Future perf saved: %s", md_path)
    return md_path, csv_path

