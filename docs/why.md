# WHY：为什么要改成单文件架构？

## 1. 旧版痛点

- 模块太多（`a_stock_analyzer/`、`modeling/`、`everyday.py`、bingwu 系列），入口复杂。
- pipeline / 模型物化分散在不同包里，阅读/调试成本高。
- UI 需要后台补算，依赖 `model_runs`、多重线程池，定位慢。
- 在“只是想快速抓数据 + 算两个模型”的场景下，大量代码属于历史负担。

## 2. 新版设计

| 组件 | 核心责任 |
| --- | --- |
| `init.py` | 建库建表（幂等），保证第一次部署时一键 OK。 |
| `getData.py` | 单一职责：调用 AkShare 拉 K 线 → 写 `stock_info/stock_daily`。 |
| `laowang.py` | 自包含指标/支撑/评分逻辑，直接产出 `stock_scores_v3` + `model_laowang_pool`。 |
| `fhkq.py` | 把旧 FHKQ 计算逻辑挪到同一个文件，按交易日写 `model_fhkq`。 |
| `everyday.py` | 自动串起 getData → laowang → fhkq，可单独跑，也可被 UI 调度。 |
| `ui.py` | 只读 Web UI，所有数据来自数据库（含 tags），并负责 15:05 自动触发 everyday。 |

### 数据流

```
AkShare → getData.py → stock_daily ┐
                               ├→ laowang.py → stock_scores_v3 / model_laowang_pool
                               └→ fhkq.py    → model_fhkq
                                          ↓
                               ui.py 读取 model_* 表展示
```

## 3. 为什么是“单文件”

- 便于部署：每个阶段只需关注一个脚本，对应日志/进度条一目了然。
- 便于排错：看到 `getData.py` 的日志就知道是抓数问题，不会混入别的逻辑。
- 便于迁移：需要调整评分规则时直接改 `laowang.py`，不用跨多模块。
- recycle_bin 中保留旧实现，必要时可以对照。

## 4. 线程与性能

- `getData.py`/`laowang.py`/`fhkq.py` 都是“外层按股票或日期并发”，内部算法尽量无副作用。
- 进度条用 `tqdm`，多线程异常会直接输出日志，方便观察瓶颈。
- MySQL/SQLite 统一走 SQLAlchemy，`init.py` 负责建表，脚本里只关注业务。

## 5. 接口与 UI

- UI 只暴露 `GET /api/dates` / `status` / `model/<name>`，无写操作，避免权限/安全问题。
- LAOWANG 表直接解析 `status_tags`，无需再依赖 JSON 字符串。
- 如果需要扩展（例如导出 CSV），可以再加一个简单脚本读取 `model_*` 表即可。
- UI 自带后台线程，每日 15:05 调用 `everyday.py`，因此“跑数”与“观看”同一进程内即可完成。

## 6. recycle_bin 的意义

- 历史模块（包括旧 pipeline、bingwu、windows 脚本等）全部迁入 `recycle_bin/`，既不污染主逻辑，又可追溯。
- 如需恢复旧功能，只要到对应目录复制出来改造即可。

总之：把原本“平台级”的架构缩成“几个命令行脚本 + 一个轻量 UI”，便于个人使用与交接，也能更快定位问题。
