@echo off
setlocal EnableExtensions EnableDelayedExpansion

set CONFIG=config.ini
if not "%~1"=="" set CONFIG=%~1

echo [init] Using config: %CONFIG%

REM init.py will CREATE DATABASE IF NOT EXISTS (requires account with create privilege).
REM sql/schema_mysql.sql remains available if you prefer to run pure SQL manually.

python init.py --config %CONFIG%

echo [init] OK
