# MySQL 多线程更新 + 回测接口

这份文档是给 MySQL 场景的快捷说明；更完整的命令说明请看根目录 `document.md`。

## 1) 配置（推荐用 config.ini）

编辑项目根目录 `config.ini`：

- 写完整连接串（推荐）：

```ini
[database]
db_url = mysql+pymysql://user:password@127.0.0.1:3306/astock?charset=utf8mb4
```

或：

- 填写 mysql 段（程序自动拼接）：

```ini
[mysql]
host = 127.0.0.1
port = 3306
user = your_user
password = your_password
database = astock
charset = utf8mb4
```

## 2) 初始化表结构

```powershell
python astock_analyzer.py init-db
```

## 3) 多线程更新（抓取 + 入库并发）

```powershell
python astock_analyzer.py run --workers 16 --start-date 20000101 --end-date 20260107
```

说明：
- 首次全量才需要较早的 `--start-date`；后续会按 `MAX(date)+1` 自动增量。
- SQLite 会自动强制 `workers=1`；MySQL 才能真正并发写入。

## 4) 回测接口

```powershell
python astock_analyzer.py backtest --nd 50 --ne 20 --k 80 --seed 42 --workers 16 --out-dir output
```

说明：
- 抽样交易日来自 `stock_daily` 的交易日序列，并自动避开末尾 `ne` 天。
- 如果 `stock_scores_v3` 有该日期的预计算评分则直接使用；否则会即时计算评分（`nd` 很大时会更慢）。

输出：
- `output/backtest_*.csv`：明细
- `output/backtest_*.md`：汇总报告

## 5) 信号后表现（future-perf）

默认统计 `ne=5,10,20,30`，并写入 `stock_future_perf`：

```powershell
python astock_analyzer.py future-perf --ne 5,10,20,30 --min-score 80 --workers 16 --out-dir output
```

## 6) Docker（可选）

如果你不想安装本地 MySQL，也可以用 `docker-compose.mysql.yml` 起一个临时实例。
