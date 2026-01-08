# -*- coding: utf-8 -*-

from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus


DEFAULT_SQLITE_PATH = "data/stock.db"
DEFAULT_CONFIG_PATH = Path("config.ini")


@dataclass(frozen=True)
class MySQLConfig:
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = ""
    password: str = ""
    database: str = ""
    charset: str = "utf8mb4"


@dataclass(frozen=True)
class AppConfig:
    db_url: Optional[str] = None
    mysql: MySQLConfig = MySQLConfig()


def load_config(path: Path) -> AppConfig:
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


def build_mysql_db_url(cfg: MySQLConfig) -> Optional[str]:
    if not cfg.user or not cfg.database:
        return None

    user = quote_plus(cfg.user)
    password = quote_plus(cfg.password or "")
    auth = f"{user}:{password}" if password else user
    return f"mysql+pymysql://{auth}@{cfg.host}:{int(cfg.port)}/{cfg.database}?charset={cfg.charset}"


def resolve_db_target(
    *,
    db_url_arg: Optional[str],
    db_arg: Optional[str],
    config_path: Optional[Path],
) -> str:
    if db_url_arg:
        return db_url_arg

    env_url = os.getenv("ASTOCK_DB_URL")
    if env_url and env_url.strip():
        return env_url.strip()

    if db_arg:
        return db_arg

    cfg_path = config_path
    if cfg_path is None and DEFAULT_CONFIG_PATH.exists():
        cfg_path = DEFAULT_CONFIG_PATH

    if cfg_path is not None:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = load_config(cfg_path)
        if cfg.db_url:
            return cfg.db_url
        mysql_url = build_mysql_db_url(cfg.mysql)
        if mysql_url:
            return mysql_url

    return DEFAULT_SQLITE_PATH
