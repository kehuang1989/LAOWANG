# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lowered = {c: str(c).lower() for c in cols}
    for c, lc in lowered.items():
        if any(k in lc for k in candidates):
            return c
    return None


def normalize_stock_code(raw) -> str:
    s = str(raw).strip()
    m = re.findall(r"\d{6}", s)
    if m:
        return m[-1]
    m2 = re.findall(r"\d+", s)
    if m2:
        return m2[-1].zfill(6)
    return s


def to_ak_symbol(stock_code: str) -> str:
    s = stock_code.strip().lower()
    if s.startswith(("sh", "sz", "bj")):
        return s
    if len(s) == 6 and s.isdigit():
        if s.startswith(("6", "9")):
            return f"sh{s}"
        if s.startswith(("8", "4")):
            return f"bj{s}"
        return f"sz{s}"
    return s


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])

    df = df.copy()
    # Some AkShare endpoints return date as index.
    if "date" not in df.columns and "日期" not in df.columns:
        idx_dt = pd.to_datetime(df.index, errors="coerce")
        if idx_dt.notna().mean() >= 0.8:
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"date", "日期"}:
            rename_map[c] = "date"
        elif lc in {"open", "开盘"}:
            rename_map[c] = "open"
        elif lc in {"high", "最高"}:
            rename_map[c] = "high"
        elif lc in {"low", "最低"}:
            rename_map[c] = "low"
        elif lc in {"close", "收盘"}:
            rename_map[c] = "close"
        elif lc in {"volume", "成交量"}:
            rename_map[c] = "volume"
        elif lc in {"amount", "成交额"}:
            rename_map[c] = "amount"
    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError(f"Daily DF missing date column; columns={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df


def fetch_stock_list() -> List[Tuple[str, str]]:
    import akshare as ak

    df = ak.stock_info_a_code_name()
    if df is None or df.empty:
        raise RuntimeError("AkShare returned empty stock list")

    code_col = pick_col(df, ["code", "代码"])
    name_col = pick_col(df, ["name", "名称"])
    if code_col is None:
        code_col = df.columns[0]
    if name_col is None:
        name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    out: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        code = normalize_stock_code(row[code_col])
        name = str(row[name_col]) if name_col in row else ""
        if code and code.isdigit() and len(code) == 6:
            out.append((code, name))

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for code, name in out:
        if code in seen:
            continue
        seen.add(code)
        deduped.append((code, name))
    return deduped


def fetch_daily(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    # Prefer Eastmoney history (often more stable), fallback to Sina daily.
    last_error: Optional[Exception] = None

    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
        return normalize_daily_df(df)
    except Exception as e:  # noqa: BLE001
        last_error = e

    for sym in [to_ak_symbol(stock_code), stock_code]:
        try:
            df = ak.stock_zh_a_daily(symbol=sym, start_date=start_date, end_date=end_date)
            return normalize_daily_df(df)
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue

    raise RuntimeError(f"Failed to fetch daily for {stock_code}: {last_error}")


def parse_market_cap_billion(raw) -> Optional[float]:
    """
    Parse market cap into "亿" (1e8 CNY) units.
    Accepts strings like "123.4亿", "1.2万亿", or plain numbers.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().replace(",", "")
    if not s:
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    val = float(m.group(1))

    if "万亿" in s:
        return val * 1e4
    if "亿" in s:
        return val
    if val > 1e8:
        return val / 1e8
    return val


def market_cap_score_from_billion(cap_billion: Optional[float]) -> int:
    # v3.0: core 30-100 亿
    if cap_billion is None or not np.isfinite(cap_billion):
        return 0
    cap_billion = float(cap_billion)
    if 30 <= cap_billion <= 100:
        return 10
    if 100 < cap_billion <= 150:
        return 6
    if 150 < cap_billion <= 300:
        return 3
    return 0


def fetch_float_market_cap_snapshot() -> dict[str, float]:
    """
    Return {stock_code(6 digits): float_market_cap_in_亿}.
    Uses a single "spot" query to avoid per-stock requests.
    """
    import akshare as ak

    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty:
        return {}

    code_col = pick_col(df, ["code", "代码"])
    if code_col is None:
        code_col = df.columns[0]

    cap_col = None
    for c in df.columns:
        sc = str(c)
        if "流通市值" in sc:
            cap_col = c
            break
    if cap_col is None:
        for c in df.columns:
            sc = str(c)
            if "总市值" in sc:
                cap_col = c
                break
    if cap_col is None:
        return {}

    out: dict[str, float] = {}
    for _, row in df.iterrows():
        code = normalize_stock_code(row[code_col])
        if not (code.isdigit() and len(code) == 6):
            continue
        cap_billion = parse_market_cap_billion(row[cap_col])
        if cap_billion is None:
            continue
        out[code] = float(cap_billion)
    return out


def fetch_market_cap_score(stock_code: str) -> int:
    import akshare as ak

    for sym in [to_ak_symbol(stock_code), stock_code]:
        try:
            df = ak.stock_a_indicator_lg(symbol=sym)
            if df is None or df.empty:
                continue

            col = None
            for c in df.columns:
                if "流通市值" in str(c):
                    col = c
                    break
            if col is None:
                for c in df.columns:
                    if "市值" in str(c):
                        col = c
                        break
            if col is None:
                continue

            cap_billion = parse_market_cap_billion(df[col].iloc[-1])
            if cap_billion is None:
                continue
            return market_cap_score_from_billion(cap_billion)
        except Exception:  # noqa: BLE001
            continue
    return 0

