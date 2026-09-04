# BMC Bioinformatics — 投稿系统信息清单

> **状态**：待作者核对；`[?]` 项为待确认/待补充（英文稿在本地，涉及标题/摘要/关键词请与英文稿比对）。

## 1. 稿件基础信息

| 字段 | 内容 | 状态 |
|---|---|---|
| 目标期刊 | BMC Bioinformatics | ✅ 已确认 |
| 文章类型 | Research article（研究论文） | ✅ |
| 标题（EN） | Cell line drug response prediction via atom-level pretrained-feature injection and a Gaussian mixture model pharmacophore modality | [?] 与本地英文稿核对 |
| 标题（ZH） | 基于原子级预训练特征注入与高斯混合模型药效团模态的细胞系药物响应预测 | ✅ |
| 摘要字数 | 英文约 271 词（背景/结果/结论三段式，符合 BMC 300–350 词要求） | ✅ |
| 关键词 | 6 个：药物反应预测、迁移学习、高斯混合模型、肿瘤与癌细胞系、精准医疗、图神经网络 | ✅ |

## 2. 作者信息

| 字段 | 内容 | 状态 |
|---|---|---|
| 姓名 | Zhihe Guo（郭知和） | ✅ |
| 通讯作者 | 是（Corresponding author） | ✅ |
| 单位 | Wuhan University, Wuhan, China（武汉大学） | ✅ |
| Email | calamitias@whu.edu.cn | ✅ |
| ORCID | https://orcid.org/0009-0000-0872-970X | ✅ |
| Authors' information | 省略（已在 Declarations 决定不写） | ✅ |
| Authors' contributions | Zhihe Guo 独立完成全部工作 | ✅ |

## 3. 建议审稿人（Suggested reviewers）

见 `docs/submission/cover_letter.md` 尾部候选清单。

- BMC 一般接受 3–5 位建议审稿人；投稿系统内可能要求填写机构与邮箱，请在最终确认后补全邮箱。[?]

## 4. 图表清单

| # | 内容 | 状态 |
|---|---|---|
| 图 1 | 数据划分与评估协议示意 | 待作图（§3 todo） |
| 图 2 | 整体模型架构 | 待作图 |
| 图 3 | 原子级预训练特征注入 | 待作图 |
| 图 4 | 药效团 GMM 模态预计算管线 | 待作图 |
| 图 5 | 三主基准与协议一致性对比（`docs/figures/results_bar.png`） | ✅ 已按 v3 数值重绘 |
| 表 1 | 三个独立主基准上各模型表现 | ✅ 正文 |
| 表 2 | protoGDA 四场景完整指标 | ✅ 正文 |
| 表 3 | 三模型计算开销对比（新增 5.2 节） | ✅ 正文 |
| 表 4 | GDSC1 零样本跨数据集验证（新增 5.3 节） | ✅ 正文 |
| 表 5–9 | 消融实验各表 | ✅ 正文 |

## 5. Declarations（投稿系统可能逐项询问）

- **Ethics approval and consent to participate**：不适用（无人类/动物）
- **Consent for publication**：不适用
- **Availability of data and materials**：GDSC2 via TDC（URL 已给）；代码 GitHub 链接 + 归档 DOI **待补充**（等你提供仓库 URL）
- **Competing interests**：无
- **Funding**：无
- **Authors' contributions**：已写
- **Acknowledgements**：致谢郭峰彪教授
- **LLM 使用声明**：Methods 4.8 已声明使用 DeepSeek/ChatGPT 辅助代码实现

## 6. 投稿附件 / 其他字段

| 项 | 内容 | 状态 |
|---|---|---|
| Cover Letter | `docs/submission/cover_letter.md` | ✅ 草稿完成，审稿人待确认 |
| 主稿文件（可编辑源） | `.md` / `.docx` / `.tex`（本地英文稿 `protoGDA_updated_manuscript.md`、`protoGDA_body_noabstract.tex`、`protoGDA_manuscript_updated_en.docx`） | 本地动作，上传时提供 |
| 图表源文件 | 矢量图（PDF/SVG）建议 | 待作图 |
| Code availability statement | 代码将开源，仓库链接/DOI 待补 | [?] 等 GitHub URL |
| 推荐/排除审稿人 | 见上 §3 | [?] |

## 7. 待作者提供（阻塞项）

1. **GitHub 仓库 URL** → 填入 Code Availability、Declarations、Cover Letter、提交表单。
2. **英文稿标题最终确认** → 替换 Cover Letter / submission_info 中的标题。
3. **建议审稿人最终名单与邮箱** → 从 cover_letter.md 尾部草稿确认。
4. **Zenodo/归档 DOI**（可等代码仓库建好后生成）。

## 8. 投稿后可能需要的补充

- 投稿系统要求填 **字数/图表数** 时从 §1/§4 取数。
- BMC 支持 **preprint**（如 bioRxiv）——如需可先挂预印本再投（非强制）。
