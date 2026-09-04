# Cover Letter — BMC Bioinformatics

> **状态说明（作者提交前请阅读）**
> - 本信为投稿用英文草稿，基于中文稿 `docs/paper_draft.md` 撰写。
> - 文中 `[TITLE — 请与本地英文稿核对]`、`[?]` 处为待你确认项（英文稿在本地）。
> - drug_cold 主表数值将以 v3 确定性协议口径统一（0.5305 ± 0.1842），Cover Letter 中已使用该口径，不与旧口径混用。

---

Dear Editors,

We are pleased to submit our manuscript entitled **[TITLE — 建议译名，请与本地英文稿核对]**: **"Cell line drug response prediction via atom-level pretrained-feature injection and a Gaussian mixture model pharmacophore modality"** for consideration for publication in *BMC Bioinformatics*.

## Summary of the work

Precision oncology relies on accurate prediction of how cancer cell lines respond to drug candidates, a task known as cancer drug response (CDR) prediction. Although deep dual-tower models have become the dominant architecture, two shortcomings persist. First, transferred/pretrained molecular features are usually fused only after the graph encoder has finished, so their information never participates in intra-graph message passing. Second, the influence of molecular conformation is largely ignored, and explicit multi-conformer enumeration incurs substantial computational cost.

In this manuscript we present **protoGDA**, a multimodal dual-tower graph attention network that addresses both issues:

1. **Atom-level gated broadcast of pretrained features.** Instead of late concatenation, each atom learns a gate that extracts a personalized subspace from drug-level Morgan/ChemBERTa features; the gated features enter the message passing of every GNN layer.
2. **A pharmacophore Gaussian mixture model (GMM) modality.** In an offline preprocessing stage we sample 30 aligned 3D conformers per drug, detect pharmacophore points, and fit a GMM over each pharmacophore instance's centroid trajectory. Rotation-invariant eigenvalues of the component covariances are injected as lightweight 18-dimensional atom-level features, capturing conformational flexibility at negligible inference overhead.
3. **Equivariant, attention-enhanced graph evolution.** The drug tower uses an EGNN backbone with equivariant graph attention (EQGAT), evolved through three cross-attention layers against a static cell-line probe tower.

## Main results

On GDSC2, under a deterministic 6-fold cross-validation protocol, protoGDA attains a CV Pearson correlation of **0.9344 ± 0.0016** (interpolation) and **0.8735 ± 0.0036** (cell-line cold-start), matching or slightly exceeding the best published/reproduced baselines. The largest gains are observed in the practically important **drug cold-start** scenario, where protoGDA reaches **0.5305 ± 0.1842** and clearly outperforms the reproduced CANDELA and MGATAF baselines run under the identical deterministic protocol. To probe generalization beyond a single assay batch, we further performed a strict zero-shot cross-dataset evaluation on **GDSC1**, an independent early-measurement batch that shares 62 drugs with GDSC2. For each shared drug we used only the fold checkpoint for which that drug was entirely held out during training and predicted its GDSC1 responses (46,993 pairs in total). protoGDA attains a pooled Pearson of **0.520** — nearly identical to its within-GDSC2 drug cold-start result — and clearly exceeds MGATAF (0.402) and CANDELA (0.198), indicating that the learned ranking of unseen drugs transfers across independent measurements rather than reflecting memorization of a single batch. Ablations show that both the atom-level feature injection and the pharmacophore GMM modality require architectures able to consume them (GINE/EGNN rather than plain GAT), and that the static cell tower is sufficient while the drug side is the performance bottleneck. To our knowledge, protoGDA is the first work to model the conformational space of pharmacophore points with a GMM and to feed the resulting rotation-invariant atom-level features into a discriminative dual-tower CDR model.

## Fit with BMC Bioinformatics

The manuscript combines cheminformatics (pharmacophore detection, conformer ensembles, Gaussian mixtures), graph neural network methodology, and a practically motivated evaluation (drug cold-start generalization). We believe it fits the scope of *BMC Bioinformatics* as an open-access methods paper with reproducible protocols, precomputed offline features, and lightweight inference.

## Declarations

- This manuscript has not been published previously and is not under consideration elsewhere.
- All authors have read and approved the final manuscript and agree to its submission.
- The authors declare no competing interests.
- No specific funding was received for this work.
- Source code and data-preparation pipelines will be made publicly available upon acceptance (GitHub repository link and archival DOI to be provided).

Thank you for your consideration. We look forward to your response.

Sincerely,

Zhihe Guo (corresponding author)
Wuhan University, Wuhan, China
Email: calamitias@whu.edu.cn
ORCID: https://orcid.org/0009-0000-0872-970X

---

## Suggested reviewers (草案 — 待作者确认)

> 以下候选基于与稿件方法的邻近性提出，供投稿系统填写；**最终名单与顺序请作者确认后再提交**。建议避免与作者存在合作关系或明显利益冲突的学者。

1. **Ewa Szczurek** — University of Warsaw, Poland. Expert in generative/probabilistic models for drug sensitivity (VADEERS, GMM priors for cancer drug response). Closest prior work on GMM in the drug-sensitivity setting.
2. **Aixia Yan** — Beijing University of Chemical Technology, China. Expert in pharmacophore concepts integrated into graph neural networks (RG-MPNN) for chemical property prediction.
3. **Saeed Dehghan / MGATAF group** — 或推荐 BMC Bioinformatics 2025 上 MGATAF 论文的通讯作者，熟悉 multi-channel graph attention for CDR（注意：MGATAF 为稿件直接对比基线，是否推荐请作者斟酌，避免编辑认为存在冲突）。
4. **Axelrod / Gómez-Bombarelli group (MIT)** — authors of the conformer-ensemble machine learning work cited in the manuscript; strong on 3D conformation modeling.
5. （备选）**Aixia Yan 之外的国内 CDR/多组学方向学者**，例如近期在 GDSC/CCLE 上做几何增强 DRP 的通讯作者（Hi-GeoMVP 通讯作者 Chen Y / Zhang L），可平衡方法学与药物-细胞系数据侧审稿。

> 提示：BMC 投稿系统建议提供 3–5 位建议审稿人。建议优先 1、2、4，并补充一位偏细胞系数据/生物学方向的审稿人以平衡视角。
