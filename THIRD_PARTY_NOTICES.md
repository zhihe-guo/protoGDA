# Third-Party Notices

本仓库以"保留所有权利"方式发布（未附带开源许可）。以下说明仅用于澄清第三方
代码/权重的归属与出处，不构成任何再分发授权。除 `pretrained/` 下另有说明的
文件外，本仓库不打包任何无开源许可的第三方数据。

`scripts/cv_runner.py` 中 scaffold-blind 划分协议参照
TransCDR（Xia et al., BMC Biology 2024）的 Murcko 骨架划分方法；TransCDR 官方
代码（GitHub / Zenodo）未提供开源许可，本仓库**不包含**其代码。配置名仍为
`scaffold_cold`，论文术语为 scaffold-blind。

## `pretrained/` 权重

`pretrained/` 下的 `.pt` 文件为本工作已完成训练的补充检查点（保留所有权利，见仓库
顶部说明）。它们不是 Zenodo 归档，也不能单独重绘全部图表。各目录 `cv_results.json`
的折均值与下列已报告的交叉验证均值相对应：

| 目录 | 对应已报告结果 |
|---|---|
| `protoGDA_interpolation/` | 表 1/2 unblinded CV（0.9344 ± 0.0016） |
| `protoGDA_cell_cold/` | 表 1/2 cell-line-blind CV（0.8735 ± 0.0036） |
| `protoGDA_drug_cold/` | 表 1/2 drug-blind CV（0.5305 ± 0.1842）与表 4 GDSC1 外部验证 |
| `protoGDA_scaffold_cold/` | 表 2 scaffold-blind CV（0.5067 ± 0.1035） |

`external_gdsc1_protoGDA_results.json` 为 GDSC1 零样本外部验证结果。

## 其它第三方资源

| 资源 | 出处 | 许可 |
|---|---|---|
| `scripts/fonts/NotoSansSC.ttf`（含 Bold） | Google Noto Fonts | SIL Open Font License 1.1（© Google；license 全文见 https://openfontlicense.org ） |
