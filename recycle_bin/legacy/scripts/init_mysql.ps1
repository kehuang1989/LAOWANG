Param(
  [string]$Config = "config.ini"
)

$ErrorActionPreference = "Stop"

Write-Host "[init] Using config: $Config"

# init.py will CREATE DATABASE IF NOT EXISTS (account must have create privilege).
# sql/schema_mysql.sql remains available if you prefer running plain SQL manually.

python init.py --config $Config

Write-Host "[init] OK"
