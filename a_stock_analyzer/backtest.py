# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from . import db
from .indicators import calc_indicators
from .levels import calc_levels
from .scoring_v3 import calc_score_v3
from .settings import Settings


@dataclass(frozen=True)
class BacktestConfig:
    nd: int
    ne: int
    k: float = 80.0
    seed: Optional[int] = None
    workers: int = 8


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_trading_days(engine) -> list[str]:
    with engine.connect() as conn:
        days = [r[0] for r in conn.execute(text("SELECT DISTINCT date FROM stock_daily ORDER BY date"))]
    return [str(d) for d in days if d]


def _load_score_days(engine) -> list[str]:
    with engine.connect() as conn:
        days = [r[0] for r in conn.execute(text("SELECT DISTINCT score_date FROM stock_scores_v3 ORDER BY score_date"))]
    return [str(d) for d in days if d]


def _sample_days(
    engine,
    nd: int,
    ne: int,
    seed: Optional[int],
) -> tuple[list[str], str]:
    trading_days = _load_trading_days(engine)
    if not trading_days:
        raise RuntimeError("No trading days found in stock_daily; run pipeline first.")
    if len(trading_days) <= ne:
        raise RuntimeError(f"Not enough trading days in stock_daily: days={len(trading_days)} <= ne={ne}")

    valid_trading = trading_days[:-ne]
    score_days = _load_score_days(engine)
    valid_trading_set = set(valid_trading)
    valid_score_days = [d for d in score_days if d in valid_trading_set]

    rng = random.Random(seed)

    if len(valid_score_days) >= nd:
        picked = rng.sample(valid_score_days, nd)
        picked.sort()
        return picked, "stock_scores_v3"

    if nd > len(valid_trading):
        raise RuntimeError(f"nd too large: nd={nd} > available_days={len(valid_trading)}")

    picked = rng.sample(valid_trading, nd)
    picked.sort()
    return picked, "stock_daily"


def _scores_exist_for_date(engine, score_date: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM stock_scores_v3 WHERE score_date = :d LIMIT 1"),
            {"d": score_date},
        ).fetchone()
    return row is not None


def _load_market_cap_score_map(engine) -> dict[str, int]:
    with engine.connect() as conn:
        latest = conn.execute(text("SELECT MAX(score_date) FROM stock_scores_v3")).fetchone()[0]
        if not latest:
            return {}
        rows = conn.execute(
            text("SELECT stock_code, market_cap_score FROM stock_scores_v3 WHERE score_date = :d"),
            {"d": str(latest)},
        ).fetchall()
    out: dict[str, int] = {}
    for code, s in rows:
        if not code:
            continue
        try:
            out[str(code)] = int(float(s)) if s is not None else 0
        except Exception:  # noqa: BLE001
            out[str(code)] = 0
    return out


def _load_all_stocks(engine) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT s.stock_code, COALESCE(i.name, '')
                FROM (SELECT DISTINCT stock_code FROM stock_daily) s
                LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                ORDER BY s.stock_code
                """
            ),
        ).fetchall()
    return [(str(code), str(name)) for code, name in rows if code]


def _evaluate_one(engine, stock_code: str, name: str, score_date: str, score: float, ne: int) -> Optional[dict]:
    with engine.connect() as conn:
        row0 = conn.execute(
            text("SELECT close FROM stock_daily WHERE stock_code = :c AND date = :d"),
            {"c": stock_code, "d": score_date},
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
            {"c": stock_code, "d": score_date, "ne": int(ne)},
        ).fetchall()

    if len(rows) < ne:
        return None

    highs = []
    lows = []
    closes = []
    for _, high, low, close in rows:
        c = float(close) if close is not None else None
        if c is None:
            continue
        closes.append(c)
        highs.append(float(high) if high is not None else c)
        lows.append(float(low) if low is not None else c)

    if len(closes) < ne:
        return None

    high_price = max(highs)
    low_price = min(lows)
    final_price = closes[-1]
    end_date = rows[-1][0]

    return {
        "score_date": score_date,
        "stock_code": stock_code,
        "name": name,
        "score": float(score),
        "score_source": "precomputed",
        "start_close": start_close,
        "end_date": end_date,
        "high_price": high_price,
        "high_return": high_price / start_close - 1.0,
        "low_price": low_price,
        "low_return": low_price / start_close - 1.0,
        "final_price": final_price,
        "final_return": final_price / start_close - 1.0,
    }


def _process_stock_for_days(
    engine,
    *,
    stock_code: str,
    name: str,
    score_days: list[str],
    ne: int,
    k: float,
    settings: Settings,
    mcap_score_map: dict[str, int],
    hist_limit: int,
) -> list[dict]:
    if not score_days:
        return []

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text(
                    """
                    SELECT date, open, high, low, close, volume, amount
                    FROM stock_daily
                    WHERE stock_code = :c
                    ORDER BY date ASC
                    """
                ),
                conn,
                params={"c": stock_code},
            )
    except Exception:  # noqa: BLE001
        logging.exception("Failed loading daily for %s", stock_code)
        return []

    if df is None or df.empty:
        return []

    df["date"] = df["date"].astype(str)
    if len(df) < (150 + ne):
        return []

    try:
        df_ind_full = calc_indicators(df)
    except Exception:  # noqa: BLE001
        logging.exception("Failed calc_indicators for %s", stock_code)
        return []

    dates = df["date"].tolist()
    date_to_idx = {d: i for i, d in enumerate(dates)}

    mcap = float(mcap_score_map.get(stock_code, 0))
    out: list[dict] = []

    for d in score_days:
        idx = date_to_idx.get(d)
        if idx is None:
            continue
        if idx + ne >= len(df):
            continue
        if idx + 1 < 150:
            continue

        hist_start = max(0, idx - hist_limit + 1)
        df_hist = df.iloc[hist_start : idx + 1].reset_index(drop=True)
        if len(df_hist) < 150:
            continue

        df_ind = df_ind_full.iloc[hist_start : idx + 1].reset_index(drop=True)

        df_level_hist = df_hist.tail(settings.level_lookback).reset_index(drop=True)
        df_level_ind = df_ind.tail(len(df_level_hist)).reset_index(drop=True)
        support, resistance, support_type, resistance_type = calc_levels(df_level_hist, df_level_ind, settings)

        (
            total,
            s_trend,
            s_pullback,
            s_vp,
            s_rsi,
            s_macd,
            s_base,
            s_space,
            s_mcap,
            status_tags,
            _risk_flags,
        ) = calc_score_v3(
            df_daily=df_hist,
            df_ind=df_ind,
            support_level=support,
            resistance_level=resistance,
            settings=settings,
            market_cap_score=mcap,
        )
        if float(total) < float(k):
            continue

        try:
            start_close = float(df_hist["close"].iloc[-1])
        except Exception:  # noqa: BLE001
            continue
        if not (start_close > 0):
            continue

        future = df.iloc[idx + 1 : idx + 1 + ne]
        if len(future) < ne:
            continue

        closes = pd.to_numeric(future["close"], errors="coerce")
        if closes.isna().any():
            continue
        highs = pd.to_numeric(future["high"], errors="coerce").fillna(closes)
        lows = pd.to_numeric(future["low"], errors="coerce").fillna(closes)

        high_price = float(highs.max())
        low_price = float(lows.min())
        final_price = float(closes.iloc[-1])
        end_date = str(future["date"].iloc[-1])

        out.append(
            {
                "score_date": d,
                "stock_code": stock_code,
                "name": name,
                "score": float(total),
                "score_source": "computed",
                "start_close": start_close,
                "end_date": end_date,
                "high_price": high_price,
                "high_return": high_price / start_close - 1.0,
                "low_price": low_price,
                "low_return": low_price / start_close - 1.0,
                "final_price": final_price,
                "final_return": final_price / start_close - 1.0,
                "support_level": support,
                "resistance_level": resistance,
                "support_type": support_type,
                "resistance_type": resistance_type,
                "trend_score": s_trend,
                "pullback_score": s_pullback,
                "volume_price_score": s_vp,
                "rsi_score": s_rsi,
                "macd_score": s_macd,
                "base_structure_score": s_base,
                "space_score": s_space,
                "market_cap_score": s_mcap,
                "status_tags": status_tags,
            }
        )

    return out


def _write_report(
    md_path: Path,
    csv_path: Path,
    cfg: BacktestConfig,
    picked_days: list[str],
    picked_source: str,
    df: pd.DataFrame,
) -> None:
    db.ensure_parent_dir(md_path)
    total_picks = int(df.shape[0])

    if total_picks:
        summary = (
            df.groupby("score_date")
            .agg(
                n=("stock_code", "count"),
                high_return_mean=("high_return", "mean"),
                high_return_median=("high_return", "median"),
                low_return_mean=("low_return", "mean"),
                low_return_median=("low_return", "median"),
                final_return_mean=("final_return", "mean"),
                final_return_median=("final_return", "median"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["score_date", "n", "high_return_mean", "low_return_mean", "final_return_mean"])

    def to_md_table(frame: pd.DataFrame) -> str:
        if frame is None or frame.empty:
            return "_No rows matched._"
        cols = list(frame.columns)
        lines = [
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
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append("# Backtest Report")
    lines.append("")
    lines.append(f"- Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- nd (sample days): {cfg.nd}")
    lines.append(f"- ne (forward days): {cfg.ne}")
    lines.append(f"- k (score threshold): {cfg.k}")
    lines.append(f"- seed: {cfg.seed}")
    lines.append(f"- workers: {cfg.workers}")
    lines.append(f"- picked_days_source: {picked_source}")
    lines.append(f"- picked_days: {', '.join(picked_days)}")
    lines.append(f"- results_rows: {total_picks}")
    lines.append(f"- csv: {csv_path.as_posix()}")
    lines.append("")

    lines.append("## Summary by Day")
    lines.append("")
    lines.append(to_md_table(summary))
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Returns are computed as (price / start_close - 1).")
    lines.append("- Window is the next `ne` rows in `stock_daily` after `score_date` (stock-specific trading days).")
    lines.append("- If `stock_scores_v3` contains `score_date`, it will be used; otherwise scores are computed on-the-fly.")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_backtest(
    db_target: str,
    *,
    nd: int,
    ne: int,
    k: float = 80.0,
    seed: Optional[int] = None,
    workers: int = 8,
    out_dir: Path = Path("output"),
) -> tuple[Path, Path]:
    cfg = BacktestConfig(nd=nd, ne=ne, k=k, seed=seed, workers=max(1, int(workers)))
    engine = db.make_engine(
        db_target,
        pool_size=max(5, cfg.workers * 2),
        max_overflow=cfg.workers * 2,
    )
    if engine.dialect.name == "sqlite" and cfg.workers > 1:
        logging.warning("SQLite does not support concurrent reads+writes well; forcing workers=1 for backtest.")
        cfg = BacktestConfig(nd=cfg.nd, ne=cfg.ne, k=cfg.k, seed=cfg.seed, workers=1)

    logging.info("Loading trading days and sampling...")
    picked_days, picked_source = _sample_days(engine, cfg.nd, cfg.ne, cfg.seed)
    logging.info("Picked %d days from %s", len(picked_days), picked_source)

    settings = Settings(db=db_target)
    hist_limit = max(settings.indicator_lookback, settings.level_lookback) + 50
    mcap_score_map = _load_market_cap_score_map(engine)

    rows_all: list[dict] = []
    score_days_set = set(_load_score_days(engine))
    picked_precomputed = [d for d in picked_days if d in score_days_set]
    picked_compute = [d for d in picked_days if d not in score_days_set]

    if picked_precomputed:
        logging.info("Using precomputed scores for %d/%d days", len(picked_precomputed), len(picked_days))
        for d in tqdm(picked_precomputed, desc="ScoreDays", unit="day"):
            with engine.connect() as conn:
                candidates = conn.execute(
                    text(
                        """
                        SELECT s.stock_code, COALESCE(i.name, ''), s.total_score
                        FROM stock_scores_v3 s
                        LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                        WHERE s.score_date = :d AND s.total_score >= :k AND (s.status_tags IS NULL OR s.status_tags NOT LIKE '%RISK_FILTERED%')
                        ORDER BY s.total_score DESC
                        """
                    ),
                    {"d": d, "k": float(cfg.k)},
                ).fetchall()

            jobs = [(str(code), str(name), d, float(score)) for code, name, score in candidates]
            if not jobs:
                continue

            if cfg.workers <= 1:
                for code, name, d0, score in jobs:
                    r = _evaluate_one(engine, code, name, d0, score, cfg.ne)
                    if r is not None:
                        rows_all.append(r)
                continue

            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                futs = [ex.submit(_evaluate_one, engine, code, name, d0, score, cfg.ne) for code, name, d0, score in jobs]
                for f in tqdm(as_completed(futs), total=len(futs), desc=f"Eval {d}", unit="stk", leave=False):
                    try:
                        r = f.result()
                        if r is not None:
                            rows_all.append(r)
                    except Exception:  # noqa: BLE001
                        logging.exception("Backtest worker failed")

    if picked_compute:
        logging.warning(
            "No precomputed scores for %d/%d days; computing scores on-the-fly (may take long).",
            len(picked_compute),
            len(picked_days),
        )
        stocks = _load_all_stocks(engine)
        logging.info("On-the-fly universe size: %d stocks", len(stocks))

        if cfg.workers <= 1:
            for code, name in tqdm(stocks, desc="Stocks", unit="stk"):
                rows_all.extend(
                    _process_stock_for_days(
                        engine,
                        stock_code=code,
                        name=name,
                        score_days=picked_compute,
                        ne=cfg.ne,
                        k=cfg.k,
                        settings=settings,
                        mcap_score_map=mcap_score_map,
                        hist_limit=hist_limit,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                futs = [
                    ex.submit(
                        _process_stock_for_days,
                        engine,
                        stock_code=code,
                        name=name,
                        score_days=picked_compute,
                        ne=cfg.ne,
                        k=cfg.k,
                        settings=settings,
                        mcap_score_map=mcap_score_map,
                        hist_limit=hist_limit,
                    )
                    for code, name in stocks
                ]
                for f in tqdm(as_completed(futs), total=len(futs), desc="Stocks", unit="stk"):
                    try:
                        rows = f.result()
                        if rows:
                            rows_all.extend(rows)
                    except Exception:  # noqa: BLE001
                        logging.exception("Backtest worker failed")

    df = pd.DataFrame(rows_all)
    ts = _timestamp()
    csv_path = out_dir / f"backtest_{ts}.csv"
    md_path = out_dir / f"backtest_{ts}.md"
    db.ensure_parent_dir(csv_path)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    _write_report(md_path, csv_path, cfg, picked_days, picked_source, df)

    logging.info("Backtest saved: %s", md_path)
    return md_path, csv_path
