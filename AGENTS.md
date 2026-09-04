# ProjectCellQuery — Agent 协作指南

本项目的代码知识图谱已由 **codebase-memory-mcp** 索引。分析代码时请优先使用图谱工具，而不是直接用 grep/glob。

## 图谱工具（MCP server: `user-codebase-memory-mcp`）

项目名固定为 **`ProjectCellQuery`**。所有工具都需传入 `project: "ProjectCellQuery"`。

| 工具 | 用途 | 何时用 |
|------|------|--------|
| `search_graph` | 语义/结构搜索：查函数、类、变量、路由定义 | 找"某个功能在哪个函数里"——优先用这个而非 grep |
| `get_code_snippet` | 读取某个符号的源码（先 `search_graph` 拿 `qualified_name`） | 要看实现细节时 |
| `trace_path` | 调用链追踪：谁调用它 / 它调用了谁 | 做影响分析、找依赖关系时 |
| `query_graph` | Cypher 查询：复杂度、热点函数、跨模块分析 | 需要聚合/统计时 |
| `get_architecture` | 架构总览：模块、依赖、热点、边界 | 刚开始理解项目时 |
| `search_code` | 图谱增强的文本搜索 | 图谱搜索命不中，需要 grep 式搜索时 |
| `get_graph_schema` | 查看图节点/边类型 | 想了解图谱结构时 |

## 使用流程建议

1. 理解代码前先 `get_architecture`（或 `search_graph`），快速把握模块划分。
2. 找定义用 `search_graph(query=...)` 或 `name_pattern`，命中后用 `get_code_snippet` 看源码。
3. 评估改动影响用 `trace_path(direction="inbound")`。
4. 只有图谱搜不到时才退回 grep/glob 兜底。

## 已知信息

- **图谱已覆盖**：`src/`（核心代码）、`baselines/`（4 个基线模型：candela/graphdrp/mgataf/transcdr）、`config/`、`scripts/`、`docs/`。
- **已排除**（不索引，勿在这些目录里找代码）：`data/`、`dataset/`、`checkpoints/`、`.specstory/`、各 `__pycache__`、baselines 内部的数据目录、二进制/序列化文件（`*.pt/*.pkl/*.npz` 等）。忽略规则见项目根 `.cbmignore`。
- **注意**：`src/data/` 是**源代码模块**（dataset.py、external_data.py、preprocessing.py），不是数据目录。
- **图谱不是 git 仓库**（本项目无 `.git`），代码改动不会自动触发增量同步；如需刷新图谱，重新调用 `index_repository(repo_path="/trashpaper/ProjectCellQuery")`。

## 图谱维护

- 重新索引：调用 `index_repository`，mode 建议 `full`。
- 图谱数据存储在 `~/.cache/codebase-memory-mcp/ProjectCellQuery.db`（全局共享，多 agent 会话可复用）。
