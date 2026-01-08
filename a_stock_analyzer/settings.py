# -*- coding: utf-8 -*-

import datetime as dt
from dataclasses import dataclass, field


def today_yyyymmdd() -> str:
    return dt.date.today().strftime("%Y%m%d")


@dataclass(frozen=True)
class Settings:
    db: str
    start_date: str = "20000101"
    end_date: str = field(default_factory=today_yyyymmdd)
    workers: int = 8
    indicator_lookback: int = 260
    level_lookback: int = 120
    breakout_lookback: int = 60
    ma_support_max_distance_pct: float = 0.10
    at_support_pct: float = 0.03
    near_resistance_pct: float = 0.05
