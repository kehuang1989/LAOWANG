#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scoring_ywcx.py

阳痿次新（YWCX）评分脚本：
- 依据 docs/scoring_ywcx.md 的规则计算 stock_scores_ywcx
- 生成 model_ywcx_pool 股票池
"""

from __future__ import annotations

try:
    import sitecustomize  # noqa: F401
except Exception:
    pass

import argparse
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError


DEFAULT_DB = "data/stock.db"


@dataclass
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"


@dataclass
class AppConfig:
    db_url: Optional[str] = None
    mysql: MySQLConfig = field(default_factory=MySQLConfig)


def load_config(path: Path) -> AppConfig:
    import configparser

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    db_url = parser.get("database", "db_url", fallback=None)
    db_url = db_url.strip() if db_url else None
    mysql = MySQLConfig(
        host=parser.get("mysql", "host", fallback="127.0.0.1").strip() or "127.0.0.1",
        port=parser.getint("mysql", "port", fallback=3306),
        user=parser.get("mysql", "user", fallback="").strip(),
        password=parser.get("mysql", "password", fallback=""),
        database=parser.get("mysql", "database", fallback="").strip(),
        charset=parser.get("mysql", "charset", fallback="utf8mb4").strip() or "utf8mb4",
    )
    return AppConfig(db_url=db_url, mysql=mysql)


def build_mysql_url(cfg: MySQLConfig) -> Optional[str]:
    if not (cfg.user and cfg.database):
        return None
    from urllib.parse import quote_plus

    user = quote_plus(cfg.user)
    password = quote_plus(cfg.password or "")
    auth = f"{user}:{password}" if password else user
    return f"mysql+pymysql://{auth}@{cfg.host}:{int(cfg.port)}/{cfg.database}?charset={cfg.charset}"


def resolve_db_target(args: argparse.Namespace) -> str:
    if getattr(args, "db_url", None):
        return str(args.db_url)
    import os

    env = os.getenv("ASTOCK_DB_URL")
    if env and env.strip():
        return env.strip()
    if getattr(args, "db", None):
        return str(args.db)
    cfg_path = getattr(args, "config", None)
    cfg_file = Path(cfg_path) if cfg_path else Path("config.ini")
    if cfg_file.exists():
        cfg = load_config(cfg_file)
        if cfg.db_url:
            return cfg.db_url
        url = build_mysql_url(cfg.mysql)
        if url:
            return url
    return DEFAULT_DB


def make_engine(db_target: str, workers: int) -> Engine:
    pool_size = max(5, min(64, workers * 2))
    max_overflow = max(10, min(128, workers * 2))
    connect_args = {}
    if "://" not in db_target and db_target.endswith(".db"):
        db_path = Path(db_target).expanduser().resolve()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_target = f"sqlite:///{db_path.as_posix()}"
    if db_target.startswith("sqlite:///"):
        connect_args["check_same_thread"] = False
    engine = create_engine(
        db_target,
        pool_pre_ping=True,
        pool_recycle=3600,
        pool_size=pool_size,
        max_overflow=max_overflow,
        connect_args=connect_args,
    )
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA journal_mode = WAL")
            cur.close()
    return engine


@dataclass(frozen=True)
class Settings:
    min_history_rows: int = 20
    max_ipo_days: int = 60
    violent_gain_threshold: float = 0.09
    violent_3day_gain: float = 0.20
    violent_volume_ratio: float = 2.0
    high_distance_min: float = 0.25
    ma10_buffer: float = 0.03
    attention_window: int = 10
    micro_trend_lookback: int = 5


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def enrich_history(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    close = df["close"]
    df["ma5"] = close.rolling(5).mean()
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma10"] = df["volume"].rolling(10).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["amt_ma20"] = df["amount"].rolling(20).mean()
    df["atr14"] = calc_atr(df["high"], df["low"], close, 14)
    return df


def _recent(series: pd.Series, idx: int = -1) -> Optional[float]:
    if series.empty:
        return None
    try:
        val = float(series.iloc[idx])
    except (IndexError, ValueError, TypeError):
        return None
    if np.isnan(val):
        return None
    return val


def load_history(engine: Engine, stock_code: str, end_date: str, min_rows: int) -> pd.DataFrame:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT date, open, high, low, close, volume, amount
                FROM stock_daily
                WHERE stock_code = :c AND date <= :e
                ORDER BY date
                """
            ),
            {"c": stock_code, "e": end_date},
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) < min_rows:
        return pd.DataFrame()
    return enrich_history(df)


def fetch_trade_dates(engine: Engine, start_date: str, end_date: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT date FROM stock_daily WHERE date BETWEEN :s AND :e ORDER BY date"),
            {"s": start_date, "e": end_date},
        ).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def list_stock_codes(engine: Engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT stock_code FROM stock_daily")).fetchall()
    return [str(r[0]) for r in rows if r and r[0]]


def _try_fetch_float_caps(engine: Engine, column: str) -> Dict[str, float]:
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"SELECT stock_code, {column} FROM stock_info")).fetchall()
    except SQLAlchemyError:
        return {}
    caps: Dict[str, float] = {}
    for code, value in rows:
        if code is None or value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        if val > 10000:
            val = val / 1e8  # assume是 “元”，转成“亿元”
        caps[str(code)] = val
    return caps


def load_float_cap_map(engine: Engine) -> Dict[str, float]:
    for col in ["float_cap", "float_cap_billion", "float_market_cap", "float_market_cap_billion"]:
        caps = _try_fetch_float_caps(engine, col)
        if caps:
            logging.info("[ywcx] 使用 stock_info.%s 作为流通市值（亿元）", col)
            return caps
    return {}


def parse_date_arg(value: str, *, default: Optional[str] = None) -> str:
    v = (value or default or "").strip()
    if len(v) == 8 and v.isdigit():
        return f"{v[0:4]}-{v[4:6]}-{v[6:8]}"
    try:
        return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid date: {value}") from exc


def _calc_listing_days(df_slice: pd.DataFrame) -> int:
    return int(len(df_slice))


def _calc_issue_price(df_slice: pd.DataFrame) -> Optional[float]:
    for col in ["close", "open"]:
        series = pd.to_numeric(df_slice[col], errors="coerce")
        val = _recent(series, idx=0)
        if val and val > 0:
            return float(val)
    return None


def _max_rolling_return(close: pd.Series, window: int) -> float:
    if len(close) <= window:
        return 0.0
    pct = close / close.shift(window) - 1
    return float(pct.max(skipna=True) or 0.0)


def hard_filter(df_slice: pd.DataFrame, settings: Settings) -> Tuple[bool, Dict[str, float]]:
    stats: Dict[str, float] = {}
    days = _calc_listing_days(df_slice)
    if days == 0 or days > settings.max_ipo_days:
        return False, stats
    issue_price = _calc_issue_price(df_slice)
    if issue_price is None or issue_price <= 0:
        return False, stats
    high_ratio = float(df_slice["high"].max(skipna=True) / issue_price) if issue_price else 0.0
    max_5day = _max_rolling_return(pd.to_numeric(df_slice["close"], errors="coerce"), 5)
    overhype = high_ratio >= 1.8 or max_5day >= 0.6
    if overhype:
        return False, stats
    stats.update(
        {
            "issue_price": float(issue_price),
            "ipo_high": float(df_slice["high"].max(skipna=True)),
            "ipo_low": float(df_slice["low"].min(skipna=True)),
            "listing_days": days,
        }
    )
    return True, stats


def risk_filters(df_slice: pd.DataFrame, stats: Dict[str, float], settings: Settings) -> bool:
    recent = df_slice.tail(5)
    if len(recent) < 2:
        return True
    close = pd.to_numeric(recent["close"], errors="coerce")
    pct_change = close.pct_change()
    if (pct_change >= settings.violent_gain_threshold).any():
        return True
    recent3 = close.tail(3)
    if len(recent3) == 3 and recent3.iloc[-1] / recent3.iloc[0] - 1 >= settings.violent_3day_gain:
        return True
    vol = pd.to_numeric(recent["volume"], errors="coerce")
    vol_ma10 = _recent(df_slice["vol_ma10"])
    if vol_ma10 and _recent(recent["volume"]) and _recent(recent["volume"]) >= settings.violent_volume_ratio * vol_ma10:
        return True
    ipo_high = stats.get("ipo_high") or 0.0
    latest_close = _recent(df_slice["close"])
    if ipo_high and latest_close:
        distance = (ipo_high - latest_close) / ipo_high
        if distance < settings.high_distance_min:
            return True
    return False


def weak_position_score(df_slice: pd.DataFrame, stats: Dict[str, float]) -> Tuple[float, bool, bool]:
    issue_price = stats.get("issue_price") or 0.0
    ipo_low = stats.get("ipo_low") or 0.0
    latest_close = _recent(df_slice["close"])
    if not latest_close or issue_price <= 0 or ipo_low <= 0:
        return 0.0, False, False
    broken = latest_close < issue_price
    dist_low = (latest_close - ipo_low) / ipo_low
    near_low = dist_low <= 0.15
    if broken and dist_low <= 0.10:
        return 10.0, True, near_low
    if broken and dist_low <= 0.20:
        return 6.0, True, near_low
    return 0.0, False, near_low


def volume_dry_score(df_slice: pd.DataFrame) -> Tuple[float, bool]:
    ma5 = _recent(df_slice["vol_ma5"])
    ma10 = _recent(df_slice["vol_ma10"])
    ma20 = _recent(df_slice["vol_ma20"])
    cond1 = bool(ma5 and ma10 and ma5 < ma10)
    cond2 = bool(ma5 and ma20 and ma5 <= 0.7 * ma20)
    if cond1 and cond2:
        return 10.0, True
    if cond1 or cond2:
        return 6.0, True
    return 0.0, False


def low_vol_score(df_slice: pd.DataFrame) -> Tuple[float, bool]:
    atr = _recent(df_slice["atr14"])
    close = _recent(df_slice["close"])
    if not atr or not close or close <= 0:
        return 0.0, False
    ratio = atr / close
    if ratio <= 0.025:
        return 10.0, True
    if ratio <= 0.04:
        return 6.0, True
    return 0.0, False


def micro_trend_score(df_slice: pd.DataFrame, settings: Settings) -> Tuple[float, bool]:
    recent = df_slice.tail(settings.micro_trend_lookback + 1)
    if len(recent) < settings.micro_trend_lookback + 1:
        return 0.0, False
    prev = recent.iloc[:-1]
    curr = recent.iloc[-1]
    prev_cond = bool((pd.to_numeric(prev["close"], errors="coerce") < pd.to_numeric(prev["ma5"], errors="coerce")).all())
    curr_close = float(curr["close"])
    curr_ma5 = float(curr["ma5"]) if pd.notna(curr["ma5"]) else None
    curr_ma10 = float(curr["ma10"]) if pd.notna(curr["ma10"]) else None
    if curr_ma5 is None or curr_close is None:
        return 0.0, False
    above_ma5 = curr_close > curr_ma5
    within_ma10 = curr_ma10 is None or curr_close <= curr_ma10 * (1 + settings.ma10_buffer)
    if prev_cond and above_ma5 and within_ma10:
        return 10.0, True
    if above_ma5:
        return 6.0, True
    return 0.0, False


def float_cap_score(stock_code: str, cap_map: Dict[str, float]) -> Tuple[float, bool]:
    cap = cap_map.get(stock_code)
    if cap is None:
        return 0.0, False
    if cap <= 30:
        return 10.0, True
    if cap <= 60:
        return 6.0, False
    if cap <= 100:
        return 3.0, False
    return 0.0, False


def attention_score(df_slice: pd.DataFrame, settings: Settings) -> float:
    recent = df_slice.tail(settings.attention_window)
    if recent.empty:
        return 0.0
    close = pd.to_numeric(recent["close"], errors="coerce")
    pct = close.pct_change().fillna(0.0)
    cond1 = (pct > 0.03).sum() <= 1
    amt = pd.to_numeric(recent["amount"], errors="coerce")
    amt_ma20 = pd.to_numeric(recent["amt_ma20"], errors="coerce")
    cond2 = bool(((amt <= 1.3 * amt_ma20) | amt_ma20.isna()).all())
    if cond1 and cond2:
        return 10.0
    if cond1 or cond2:
        return 6.0
    return 0.0


def build_tags(
    broken: bool,
    near_low: bool,
    vol_dry: bool,
    low_vol: bool,
    just_ma5: bool,
    small_float: bool,
) -> List[str]:
    tags: List[str] = []
    if broken:
        tags.append("BROKEN_IPO")
    if near_low:
        tags.append("NEAR_IPO_LOW")
    if vol_dry:
        tags.append("VOLUME_DRY")
    if low_vol:
        tags.append("LOW_VOL")
    if just_ma5:
        tags.append("JUST_ABOVE_MA5")
    if small_float:
        tags.append("SMALL_FLOAT")
    return tags


def compute_score(
    df_slice: pd.DataFrame,
    stock_code: str,
    stats: Dict[str, float],
    cap_map: Dict[str, float],
    settings: Settings,
) -> Dict[str, object]:
    w_pos, broken, near_low = weak_position_score(df_slice, stats)
    vol_dry, vol_flag = volume_dry_score(df_slice)
    low_vol, low_flag = low_vol_score(df_slice)
    micro_trend, just_ma5 = micro_trend_score(df_slice, settings)
    cap_score, small_float = float_cap_score(stock_code, cap_map)
    att_score = attention_score(df_slice, settings)
    total = (
        (w_pos / 10.0) * 20.0
        + (vol_dry / 10.0) * 20.0
        + (low_vol / 10.0) * 15.0
        + (micro_trend / 10.0) * 20.0
        + (cap_score / 10.0) * 15.0
        + (att_score / 10.0) * 10.0
    )
    tags = build_tags(broken, near_low, vol_flag, low_flag, just_ma5, small_float)
    return {
        "total_score": float(total),
        "weak_position_score": float(w_pos),
        "volume_dry_score": float(vol_dry),
        "low_volatility_score": float(low_vol),
        "micro_trend_score": float(micro_trend),
        "float_cap_score": float(cap_score),
        "attention_score": float(att_score),
        "status_tags": tags,
    }


def compute_scores_for_stock(
    *,
    engine: Engine,
    stock_code: str,
    target_dates: Sequence[str],
    cap_map: Dict[str, float],
    settings: Settings,
) -> List[Dict[str, object]]:
    df_hist = load_history(engine, stock_code, target_dates[-1], min_rows=settings.min_history_rows)
    if df_hist.empty:
        return []
    hist_dates = set(df_hist["date"].tolist())
    rows: List[Dict[str, object]] = []
    for trade_date in target_dates:
        if trade_date not in hist_dates:
            continue
        df_slice = df_hist[df_hist["date"] <= trade_date]
        eligible, stats = hard_filter(df_slice, settings)
        if not eligible:
            continue
        if risk_filters(df_slice, stats, settings):
            rows.append(
                {
                    "stock_code": stock_code,
                    "score_date": trade_date,
                    "total_score": 0.0,
                    "weak_position_score": 0.0,
                    "volume_dry_score": 0.0,
                    "low_volatility_score": 0.0,
                    "micro_trend_score": 0.0,
                    "float_cap_score": 0.0,
                    "attention_score": 0.0,
                    "status_tags": json.dumps(["RISK_FILTERED"], ensure_ascii=False),
                }
            )
            continue
        comp = compute_score(df_slice, stock_code, stats, cap_map, settings)
        comp["status_tags"] = json.dumps(comp["status_tags"], ensure_ascii=False)
        comp.update({"stock_code": stock_code, "score_date": trade_date})
        rows.append(comp)
    return rows


def upsert_rows(engine: Engine, table: str, cols: Sequence[str], rows: Sequence[Dict[str, object]], key_cols: Sequence[str]) -> None:
    if not rows:
        return
    placeholders = ", ".join([f":{c}" for c in cols])
    col_list = ", ".join(cols)
    if engine.dialect.name == "sqlite":
        stmt = f"INSERT OR REPLACE INTO {table}({col_list}) VALUES({placeholders})"
    else:
        updates = ", ".join([f"{c}=VALUES({c})" for c in cols if c not in key_cols])
        stmt = f"INSERT INTO {table}({col_list}) VALUES({placeholders}) ON DUPLICATE KEY UPDATE {updates}"
    with engine.begin() as conn:
        conn.execute(text(stmt), rows)


def delete_by_trade_date(engine: Engine, table: str, trade_date: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE trade_date = :d"), {"d": trade_date})


def build_pool(engine: Engine, trade_date: str, top_n: int, min_score: float) -> None:
    delete_by_trade_date(engine, "model_ywcx_pool", trade_date)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT s.stock_code,
                       COALESCE(i.name, '') AS stock_name,
                       d.close AS close,
                       s.total_score,
                       s.weak_position_score,
                       s.volume_dry_score,
                       s.low_volatility_score,
                       s.status_tags
                FROM stock_scores_ywcx s
                LEFT JOIN stock_info i ON i.stock_code = s.stock_code
                LEFT JOIN stock_daily d ON d.stock_code = s.stock_code AND d.date = s.score_date
                WHERE s.score_date = :d AND s.total_score >= :min_score
                ORDER BY s.total_score DESC, s.weak_position_score DESC
                LIMIT :lim
                """
            ),
            {"d": trade_date, "min_score": float(min_score), "lim": max(top_n * 3, top_n)},
        ).fetchall()
    rows = rows[:top_n]
    payload: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        payload.append(
            {
                "trade_date": trade_date,
                "rank_no": idx,
                "stock_code": row[0],
                "stock_name": row[1],
                "close": row[2],
                "total_score": row[3],
                "weak_position_score": row[4],
                "volume_dry_score": row[5],
                "low_volatility_score": row[6],
                "status_tags": row[7],
            }
        )
    if payload:
        upsert_rows(
            engine,
            "model_ywcx_pool",
            [
                "trade_date",
                "rank_no",
                "stock_code",
                "stock_name",
                "close",
                "total_score",
                "weak_position_score",
                "volume_dry_score",
                "low_volatility_score",
                "status_tags",
            ],
            payload,
            ["trade_date", "stock_code"],
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="阳痿次新（YWCX）评分脚本")
    parser.add_argument("--config", default=None)
    parser.add_argument("--db-url", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--top", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=55.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    start = parse_date_arg(args.start_date)
    end = parse_date_arg(args.end_date)
    if start > end:
        raise SystemExit("start-date must be <= end-date")

    db_target = resolve_db_target(args)
    workers = max(1, int(args.workers))
    engine = make_engine(db_target, workers)

    trade_dates = fetch_trade_dates(engine, start, end)
    if not trade_dates:
        logging.warning("指定区间内没有交易日数据：%s ~ %s", start, end)
        return 0

    stock_codes = list_stock_codes(engine)
    if not stock_codes:
        logging.warning("stock_daily 为空，无法计算")
        return 0

    cap_map = load_float_cap_map(engine)
    settings = Settings()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    score_rows_all: List[Dict[str, object]] = []
    progress = tqdm(total=len(stock_codes), desc="YWCX评分", unit="stock")

    def worker(code: str) -> List[Dict[str, object]]:
        return compute_scores_for_stock(engine=engine, stock_code=code, target_dates=trade_dates, cap_map=cap_map, settings=settings)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(worker, code): code for code in stock_codes}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
                score_rows_all.extend(rows)
            except Exception as exc:  # noqa: BLE001
                logging.exception("[ywcx] 计算 %s 失败: %s", futs[fut], exc)
            finally:
                progress.update(1)
    progress.close()

    logging.info("写入 stock_scores_ywcx: %d 行", len(score_rows_all))
    upsert_rows(
        engine,
        "stock_scores_ywcx",
        [
            "stock_code",
            "score_date",
            "total_score",
            "weak_position_score",
            "volume_dry_score",
            "low_volatility_score",
            "micro_trend_score",
            "float_cap_score",
            "attention_score",
            "status_tags",
        ],
        score_rows_all,
        ["stock_code", "score_date"],
    )

    pool_progress = tqdm(total=len(trade_dates), desc="生成YWCX股票池", unit="day")
    for trade_date in trade_dates:
        try:
            build_pool(engine, trade_date, top_n=int(args.top), min_score=float(args.min_score))
        except Exception as exc:  # noqa: BLE001
            logging.exception("生成 %s 股票池失败: %s", trade_date, exc)
        finally:
            pool_progress.update(1)
    pool_progress.close()

    logging.info("YWCX 完成：dates=%d stocks=%d", len(trade_dates), len(stock_codes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
