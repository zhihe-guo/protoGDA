# TODO / 待办清单

> 更新时间：当前状态
> 状态标记：`[ ]` 待办，`[x]` 已完成，`[?]` 需要作者确认

## 1. 作者与元信息

- [x] 确认中文姓名：郭知和（Zhihe Guo）
- [x] 确认单位：武汉大学
- [x] 补充通讯邮箱：calamitias@whu.edu.cn
- [x] 补充 ORCID：https://orcid.org/0009-0000-0872-970X
- [x] 确认使用 DeepSeek/ChatGPT 辅助写作，并已在 Methods 中声明

## 2. 数据与代码可用性

- [x] 补充 GDSC2 在 TDC 的 URL：https://tdcommons.ai/multi_pred_tasks/drugres/
- [x] 补充数据集引用：Iorio 2016 [1]；Huang/TDC 2021 [15]
- [ ] 补充 GitHub 仓库 URL（**待作者提供**；拿到后我会补齐 Code Availability、README 状态与文件清单）
- [ ] 补充 Zenodo/归档 DOI（**待作者提供**，可在 GitHub 建仓后一键归档）
- [ ] 补充代码的 OS、编程语言、License、使用限制（OS=Linux、Python≥3.11、依赖见 requirements.txt 可先填；License 与仓库 URL 一起确认，默认建议 MIT）
- [ ] 将 Code Availability 从“即将发布”改为可访问状态

## 3. 图表

- [ ] 图 1：数据划分与评估协议
- [ ] 图 2：总体模型架构
- [ ] 图 3：原子级预训练特征注入
- [ ] 图 4：药效团 GMM 模态预计算管线
- [ ] 图 5：确认高清版本与分辨率
- [ ] 所有图题 ≤15 词，图例 ≤300 词
- [ ] 所有表格标题/表例符合 BMC 要求

## 4. 格式与结构

- [x] 增加独立 Conclusions 章节
- [x] 增加 List of Abbreviations
- [x] 增加 Declarations 框架
- [x] 在引言中强化综述性总结与研究缺口
- [x] 将“基于不变量的自注意力”统一为“EQGAT 等变图注意力”
- [x] 统一 drug_cold 的 0.5305 / 0.5306 口径（以 v3 确定性协议为准：protoGDA 0.5305±0.1842；CANDELA 0.3370±0.1089、MGATAF 0.4325±0.1603 已按 v3 重跑并写入表 1 / 正文 / 图 5）
- [x] 确认摘要字数：英文约 271 词，低于 300/350 限制
- [x] 中文稿参考文献已改写为 Vancouver 风格
- [ ] 英文稿 / LaTeX 参考文献尚未改为真正的 Vancouver / OUP 格式
- [x] 双倍行距、行号、页码（BMC 模板已启用 referee+lineno）
- [x] 删除多余占位符（已扫描全文：无 `~~删除线~~`/TODO/TBD/公式占位残留；仅保留需等 GitHub URL 的"仓库链接与归档 DOI 待补充"）

## 5. Declarations 具体内容

- [x] Ethics approval and consent to participate：不适用，已写
- [x] Consent for publication：不适用，已写
- [x] Availability of data and materials：数据 URL 已补；代码链接仍待补
- [x] Competing interests：无利益冲突，已写
- [x] Funding：没有资金赞助，已写
- [x] Authors' contributions：郭知和独立完成，已写
- [x] Acknowledgements：感谢郭峰彪教授，已写
- [x] Authors' information（可选）：已决定省略，不写入 Declarations

## 6. 内容强化

- [x] 引言增加“轻量性”贡献点
- [x] 摘要增加 GMM 轻量性描述
- [x] 建议补计算开销对比（可选）— 已完成：`scripts/measure_compute_cost.py` 测出参数量与推理时延并写入论文新 5.2 节（表 3）：protoGDA 参数量 0.91M（为 CANDELA 54% / MGATAF 25%），GMM 模态边际仅 1,152 参数（0.13%）；batch 256 推理时延 36.4ms/batch（CANDELA 15.7 / MGATAF 10.6，RTX 4090）
- [x] 建议补外部数据集验证 — 已完成：零样本跨库协议（`scripts/external_validate_gdsc1.py`，`--model cellquery|candela|mgataf`）：对每个共享药物只用 drug_cold v3 中以该药为 test 药的 checkpoint（训练未见该药），预测 GDSC1 上 62 held-out 药 × 803 共享细胞（46,993 对）；结果写入论文新 5.3 节（表 4）：pooled Pearson protoGDA **0.520** > MGATAF 0.402 > CANDELA 0.198；drug-level Pearson 0.578 > 0.436 > 0.172（结果存 `checkpoints/external_gdsc1_*_results.json`）；7.5 局限段同步更新
- [x] 讨论中进一步说明“首次将 GMM 引入 CDR”的查新依据 — 已完成（并经作者确认做引言重构）：引言现为“背景 → 任务与瓶颈 → 特征增强方法[27,9] → 药效团综述（RG-MPNN[29]/MoGraphDRP[30] 独立成段）→ DTI 构象动机段[3,10]（原第 2 段改造、前移至 GMM 段前，多构象比对并入）→ GMM 段（融合 VADEERS[28]）→ 定位与贡献”；“查新说明”子节已从讨论删除；新增参考文献 28–30

## 7. 中英文同步

- [x] 中文稿 `paper_draft.md` 已更新
- [x] 英文稿 `protoGDA_updated_manuscript.md` 已同步主要修改（后续仍建议整体校对 citation 编号与术语）
- [x] 重新生成 `protoGDA_body_noabstract.tex`
- [x] 重新编译 BMC/Bioinformatics 两个官方模板 PDF
- [x] 更新 Word 导出稿 `protoGDA_manuscript_updated_en.docx`

## 8. 投稿材料

- [x] 完成期刊选择最终确认：**BMC Bioinformatics**
- [x] 完成 Cover Letter（`docs/submission/cover_letter.md`，含候选审稿人，待作者确认）
- [x] 准备投稿系统所需信息（`docs/submission/submission_info.md` 清单）
- [ ] 上传可编辑源文件
