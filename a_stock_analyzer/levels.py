# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .settings import Settings


def volume_profile_levels(df: pd.DataFrame, bins: int = 20) -> Tuple[Optional[float], Optional[float]]:
    if df.empty or df["close"].isna().all():
        return None, None

    low = float(df["low"].min())
    high = float(df["high"].max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None, None

    typical = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(dtype=float)
    volume = df["volume"].fillna(0).to_numpy(dtype=float)

    edges = np.linspace(low, high, bins + 1)
    idx = np.digitize(typical, edges) - 1
    idx = idx.clip(0, bins - 1).astype(int)

    vol_by_bin = np.bincount(idx, weights=volume, minlength=bins).astype(float)

    mids = (edges[:-1] + edges[1:]) / 2
    current_close = float(df["close"].iloc[-1])

    support = None
    mask_support = mids < current_close
    if mask_support.any():
        support_mids = mids[mask_support]
        best = np.argmax(vol_by_bin[mask_support])
        support = float(support_mids[best])

    resistance = None
    mask_res = mids > current_close
    if mask_res.any():
        res_mids = mids[mask_res]
        best = np.argmax(vol_by_bin[mask_res])
        resistance = float(res_mids[best])

    return support, resistance


def recent_pivot_low(df: pd.DataFrame, window: int = 5) -> Optional[float]:
    if df.empty:
        return None
    low = df["low"].astype(float)
    rolling_min = low.rolling(window, center=True).min()
    pivots = df[low == rolling_min]
    if not pivots.empty:
        return float(pivots["low"].iloc[-1])
    return float(low.min())


def recent_pivot_high(df: pd.DataFrame, window: int = 5) -> Optional[float]:
    if df.empty:
        return None
    high = df["high"].astype(float)
    rolling_max = high.rolling(window, center=True).max()
    pivots = df[high == rolling_max]
    if not pivots.empty:
        return float(pivots["high"].iloc[-1])
    return float(high.max())


def calc_levels(
    df_daily: pd.DataFrame,
    df_ind: pd.DataFrame,
    settings: Settings,
) -> Tuple[Optional[float], Optional[float], str, str]:
    if df_daily.empty or df_ind.empty:
        return None, None, "", ""

    close = float(df_daily["close"].iloc[-1])
    ma60 = df_ind["ma60"].iloc[-1]
    ma120 = df_ind["ma120"].iloc[-1]

    ma_candidates = []
    if pd.notna(ma60) and float(ma60) < close:
        ma_candidates.append(("MA60", float(ma60)))
    if pd.notna(ma120) and float(ma120) < close:
        ma_candidates.append(("MA120", float(ma120)))

    nearest_ma_type = ""
    nearest_ma = None
    if ma_candidates:
        nearest_ma_type, nearest_ma = min(ma_candidates, key=lambda x: abs(close - x[1]))

    pivot_low = recent_pivot_low(df_daily)
    pivot_high = recent_pivot_high(df_daily)
    vp_support, vp_resistance = volume_profile_levels(df_daily)

    support_level = None
    support_type = ""
    if nearest_ma is not None:
        dist = (close - nearest_ma) / close
        if dist <= settings.ma_support_max_distance_pct:
            support_level = nearest_ma
            support_type = nearest_ma_type

    if support_level is None and pivot_low is not None:
        support_level = float(pivot_low)
        support_type = "前低"

    if vp_support is not None:
        if support_level is None:
            support_level = float(vp_support)
            support_type = "成交密集区"
        else:
            if abs(float(support_level) - float(vp_support)) / close <= 0.03:
                support_level = max(float(support_level), float(vp_support))
                support_type = f"{support_type}+成交密集区"

    resistance_level = None
    resistance_type = ""
    resistance_candidates = []
    if pivot_high is not None and float(pivot_high) > close:
        resistance_candidates.append(("前高", float(pivot_high)))
    if vp_resistance is not None and float(vp_resistance) > close:
        resistance_candidates.append(("成交密集区", float(vp_resistance)))
    if resistance_candidates:
        resistance_type, resistance_level = min(resistance_candidates, key=lambda x: abs(x[1] - close))

    return support_level, resistance_level, support_type, resistance_type
