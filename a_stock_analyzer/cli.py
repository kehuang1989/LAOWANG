# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from . import db
from .backtest import run_backtest
from .config import resolve_db_target
from .future_perf import run_future_perf
from .pipeline import export_pool, run_pipeline
from .settings import Settings, today_yyyymmdd


def setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(message)s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="A股中线右侧交易分析系统（SQLite + AkShare）")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    p.add_argument("--config", default=None, help="Path to config.ini (optional)")

    sub = p.add_subparsers(dest="cmd", required=True)

    s_init = sub.add_parser("init-db", help="Create tables")
    s_init.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_init.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")

    s_run = sub.add_parser("run", help="Update + calculate + score")
    s_run.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_run.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_run.add_argument("--start-date", default="20000101")
    s_run.add_argument("--end-date", default=today_yyyymmdd())
    s_run.add_argument("--workers", type=int, default=8, help="Thread workers (MySQL recommended)")
    s_run.add_argument("--limit-stocks", type=int, default=None)

    s_export = sub.add_parser("export", help="Export pool CSV from DB")
    s_export.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_export.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_export.add_argument("--output", default=f"output/pool_{today_yyyymmdd()}.csv")
    s_export.add_argument("--top", type=int, default=200)
    s_export.add_argument("--min-score", type=float, default=None)
    s_export.add_argument("--require-tags", default=None, help="Comma-separated, e.g. TREND_UP,AT_SUPPORT")
    s_export.add_argument("--min-resistance-distance", type=float, default=None, help="e.g. 0.10 for 10%")

    s_bt = sub.add_parser("backtest", help="Random day backtest based on score table")
    s_bt.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_bt.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_bt.add_argument("--nd", type=int, default=20, help="Random sample days")
    s_bt.add_argument("--ne", type=int, default=20, help="Forward trading days")
    s_bt.add_argument("--k", type=float, default=80.0, help="Score threshold")
    s_bt.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    s_bt.add_argument("--workers", type=int, default=8, help="Thread workers")
    s_bt.add_argument("--out-dir", default="output", help="Output directory")

    s_fp = sub.add_parser("future-perf", help="Signal future performance (ne=5/10/20/30)")
    s_fp.add_argument("--db", default=None, help="SQLite db path (optional)")
    s_fp.add_argument("--db-url", default=None, help="SQLAlchemy DB URL; overrides --db")
    s_fp.add_argument("--ne", default="5,10,20,30", help="Comma-separated, e.g. 5,10,20,30")
    s_fp.add_argument("--min-score", type=float, default=80.0, help="Signal threshold k")
    s_fp.add_argument("--signal-date", default=None, help="YYYY-MM-DD (optional). Default: latest eligible day per ne")
    s_fp.add_argument("--top", type=int, default=None, help="Limit signals per day (optional)")
    s_fp.add_argument("--workers", type=int, default=8, help="Thread workers")
    s_fp.add_argument("--out-dir", default="output", help="Output directory")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    db_target = resolve_db_target(
        db_url_arg=getattr(args, "db_url", None),
        db_arg=getattr(args, "db", None),
        config_path=Path(args.config) if args.config else None,
    )

    if args.cmd == "init-db":
        engine = db.make_engine(db_target, pool_size=5, max_overflow=10)
        db.init_db(engine)
        logging.info("DB initialized: %s", db_target)
        return 0

    if args.cmd == "run":
        settings = Settings(
            db=db_target,
            start_date=args.start_date,
            end_date=args.end_date,
            workers=args.workers,
        )
        run_pipeline(settings, limit_stocks=args.limit_stocks)
        return 0

    if args.cmd == "export":
        require_tags = args.require_tags.split(",") if args.require_tags else None
        export_pool(
            db_target=db_target,
            output_csv=Path(args.output),
            top_n=args.top,
            min_score=args.min_score,
            require_tags=require_tags,
            min_resistance_distance=args.min_resistance_distance,
        )
        return 0

    if args.cmd == "backtest":
        run_backtest(
            db_target=db_target,
            nd=args.nd,
            ne=args.ne,
            k=args.k,
            seed=args.seed,
            workers=args.workers,
            out_dir=Path(args.out_dir),
        )
        return 0

    if args.cmd == "future-perf":
        ne_list = [int(x.strip()) for x in str(args.ne).split(",") if x.strip()]
        run_future_perf(
            db_target=db_target,
            ne_list=ne_list,
            min_score=args.min_score,
            signal_date=args.signal_date,
            workers=args.workers,
            top_n=args.top,
            out_dir=Path(args.out_dir),
        )
        return 0

    parser.error("Unknown command")
    return 2
