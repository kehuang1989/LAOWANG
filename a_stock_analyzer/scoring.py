# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .akshare_client import fetch_market_cap_score
from .settings import Settings


def score_trend(df_daily: pd.DataFrame, df_ind: pd.DataFrame, breakout_lookback: int) -> int:
    ma20 = df_ind["ma20"].iloc[-1]
    ma60 = df_ind["ma60"].iloc[-1]
    ma120 = df_ind["ma120"].iloc[-1]
    close = float(df_daily["close"].iloc[-1])

    if pd.notna(ma20) and pd.notna(ma60) and pd.notna(ma120):
        if float(ma20) > float(ma60) > float(ma120) and close > float(ma60):
            rolling_high = df_daily["high"].rolling(breakout_lookback).max()
            if pd.notna(rolling_high.iloc[-1]) and close >= float(rolling_high.iloc[-1]):
                return 10
            return 7

        mas = np.array([float(ma20), float(ma60), float(ma120)])
        if np.isfinite(mas).all():
            if (mas.max() - mas.min()) / close <= 0.02:
                return 4

    return 0


def score_pullback(
    close: float,
    ma20: Optional[float],
    ma60: Optional[float],
    support_level: Optional[float],
    at_support_pct: float,
) -> int:
    if support_level is None or not np.isfinite(support_level):
        return 0
    if close < support_level:
        return 0

    near_support = (close - support_level) / close <= at_support_pct
    if near_support:
        if ma60 is not None and np.isfinite(ma60) and close > float(ma60):
            return 10
        return 6

    if ma20 is not None and ma60 is not None:
        if np.isfinite(ma20) and np.isfinite(ma60):
            if close < float(ma20) and close > float(ma60):
                return 6

    return 2


def score_rsi(rsi14: Optional[float]) -> int:
    if rsi14 is None or not np.isfinite(rsi14):
        return 0
    r = float(rsi14)
    if 50 <= r <= 65:
        return 10
    if 40 <= r < 50:
        return 6
    if r > 70:
        return 0
    if 30 <= r < 40:
        return 4
    return 2


def score_macd(
    diff: Optional[float],
    dea: Optional[float],
    prev_diff: Optional[float],
    prev_dea: Optional[float],
) -> int:
    if diff is None or dea is None or prev_diff is None or prev_dea is None:
        return 0
    if not all(np.isfinite([diff, dea, prev_diff, prev_dea])):
        return 0

    crossed_up = prev_diff <= prev_dea and diff > dea
    crossed_down = prev_diff >= prev_dea and diff < dea
    if crossed_down:
        return 0
    if crossed_up:
        return 10 if diff > 0 else 5
    return 5 if diff > dea else 2


def score_volume_price(df_daily: pd.DataFrame, breakout_lookback: int) -> int:
    if len(df_daily) < 30:
        return 0
    close = df_daily["close"]
    volume = df_daily["volume"].fillna(0)
    avg20 = volume.rolling(20).mean()

    rolling_high_prev = df_daily["high"].rolling(breakout_lookback).max().shift(1)
    breakout = bool(pd.notna(rolling_high_prev.iloc[-1]) and float(close.iloc[-1]) >= float(rolling_high_prev.iloc[-1]))

    score = 0
    if breakout and float(volume.iloc[-1]) >= 1.3 * float(avg20.iloc[-1] or 0):
        score += 5

    pullback_day = float(close.iloc[-1]) < float(close.iloc[-2])
    if pullback_day and float(volume.iloc[-1]) <= 0.9 * float(avg20.iloc[-1] or 0):
        score += 3

    rebound = float(close.iloc[-1]) > float(close.iloc[-2])
    if rebound and float(volume.iloc[-1]) > float(volume.iloc[-2]) and float(volume.iloc[-1]) > float(avg20.iloc[-1] or 0):
        score += 2

    heavy_selloff = float(close.iloc[-1]) < float(close.iloc[-2]) and float(volume.iloc[-1]) >= 1.5 * float(avg20.iloc[-1] or 0)
    if heavy_selloff:
        return 0

    return min(10, score) if score else 5


def build_tags(
    close: float,
    support_level: Optional[float],
    resistance_level: Optional[float],
    trend_score: int,
    pullback_score: int,
    rsi14: Optional[float],
    settings: Settings,
) -> list:
    tags = []
    if trend_score >= 7:
        tags.append("TREND_UP")
    if pullback_score >= 6:
        tags.append("PULLBACK")
    if support_level is not None and (close - support_level) / close <= settings.at_support_pct:
        tags.append("AT_SUPPORT")
    if resistance_level is not None and (resistance_level - close) / close <= settings.near_resistance_pct:
        tags.append("NEAR_RESISTANCE")
    if rsi14 is not None and np.isfinite(rsi14) and float(rsi14) > 70:
        tags.append("RISK_CHASE")
    return tags


def calc_score(
    stock_code: str,
    df_daily: pd.DataFrame,
    df_ind: pd.DataFrame,
    support_level: Optional[float],
    resistance_level: Optional[float],
    settings: Settings,
    market_cap_score: Optional[int] = None,
) -> Tuple[float, int, int, int, int, int, int, str]:
    latest = df_ind.iloc[-1]
    prev = df_ind.iloc[-2] if len(df_ind) >= 2 else latest

    close = float(df_daily["close"].iloc[-1])
    s_trend = score_trend(df_daily, df_ind, settings.breakout_lookback)
    s_pullback = score_pullback(
        close=close,
        ma20=float(latest["ma20"]) if pd.notna(latest["ma20"]) else None,
        ma60=float(latest["ma60"]) if pd.notna(latest["ma60"]) else None,
        support_level=support_level,
        at_support_pct=settings.at_support_pct,
    )
    s_vp = score_volume_price(df_daily, settings.breakout_lookback)
    s_rsi = score_rsi(float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else None)
    s_macd = score_macd(
        diff=float(latest["macd_diff"]) if pd.notna(latest["macd_diff"]) else None,
        dea=float(latest["macd_dea"]) if pd.notna(latest["macd_dea"]) else None,
        prev_diff=float(prev["macd_diff"]) if pd.notna(prev["macd_diff"]) else None,
        prev_dea=float(prev["macd_dea"]) if pd.notna(prev["macd_dea"]) else None,
    )
    s_mcap = int(market_cap_score) if market_cap_score is not None else fetch_market_cap_score(stock_code)

    total = (
        (s_trend / 10) * 30
        + (s_pullback / 10) * 20
        + (s_vp / 10) * 15
        + (s_rsi / 10) * 15
        + (s_macd / 10) * 10
        + (s_mcap / 10) * 10
    )

    tags = build_tags(
        close=close,
        support_level=support_level,
        resistance_level=resistance_level,
        trend_score=s_trend,
        pullback_score=s_pullback,
        rsi14=float(latest["rsi14"]) if pd.notna(latest["rsi14"]) else None,
        settings=settings,
    )

    return float(total), s_trend, s_pullback, s_vp, s_rsi, s_macd, s_mcap, ",".join(tags)
