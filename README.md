# protoGDA — 药效团 GMM 模态增强的多模态双塔 CDR 模型

> 开发代号 *CellQuery*。用于细胞系药物响应（Cancer Drug Response, CDR）预测：
> 给定药物分子与癌细胞系，预测药物对细胞系的药效（ln IC50）。
>
> **论文**：*Cell line drug response prediction via atom-level pretrained-feature injection and a Gaussian mixture model pharmacophore modality*（投稿中，BMC Bioinformatics）。接收后将在此处补充 DOI。

## 简介

protoGDA 是一套面向 **药物冷启动（drug cold-start）** 场景的多模态双塔模型：

1. **原子级预训练特征门控广播**：药物级 Morgan / ChemBERTa 特征经原子感知门控抽取个性化子空间，进入每一层 GNN 的消息传递（而非仅在末端拼接）。
2. **药效团高斯混合模型（GMM）模态**：离线预采样 30 个对齐 3D 构象，检测药效团位点，对其跨构象质心轨迹拟合 GMM；分量协方差特征值（旋转不变）作为轻量 18 维原子级特征。
3. **EQGAT 等变图注意力增强的 EGNN 药物塔**：与静态细胞系探针塔逐层做交叉注意力演化。

预训练特征（Morgan/ChemBERTa）、3D 构象与药效团 GMM 特征全部在 `scripts/precompute_*.py` 离线生成，训练/推理阶段零在线构象开销。论文要点见 `THIRD_PARTY_NOTICES.md` 中 `pretrained/` 一节与下列复现命令。

## 目录结构

```
src/                 模型、数据管线、训练器、评估（核心代码）
config/              各场景配置（interpolation / cell_cold / drug_cold / scaffold_cold）
baselines/           CANDELA / MGATAF / GraphDRP 的独立重写实现（详见 THIRD_PARTY_NOTICES.md）
scripts/             CV、外部验证、预计算、绘图与测量脚本
pretrained/          复现论文结果的模型权重（6 折 best.pt + cv_results.json）
data/                本地数据缓存（不入库；运行时自动生成/需手动放置，见下）
```

## 安装与数据

```bash
pip install -e ".[linux-data,dev]"        # Linux
# Windows 见原 README 注意事项（pytdc 需 --no-deps）
```

- **数据**：GDSC2 经 [TDC DrugRes](https://tdcommons.ai/multi_pred_tasks/drugres/) 加载；基因表达等特征首次运行自动下载并缓存到 `data/`。`baselines/candela`、`baselines/mgataf` 使用 `data/gdsc2.pkl`（由 TDC 导出后放置）。
- **预计算缓存**：构象、药效团 GMM、ChemBERTa 嵌入缺失时，`src/data/dataset.py` 会自动调用 `scripts/precompute_conformers.py` / `precompute_pharmacophores.py` / `precompute_chemberta.py` 生成。

## 复现论文结果

论文全部数值来自 6 折交叉验证（v3 确定性协议：固定划分 seed=42、每折 seed=42+fold、60 epoch、验证 Pearson 选 checkpoint）。

```bash
# 主模型 CV（表 1/表 2 的 protoGDA 行）
python scripts/cv_runner.py --model cellquery --k 6 --split_by drug_cold \
    --config config/cellquery_drug_cold.yaml --epochs 60 --patience 999 \
    --checkpoint-metric pearson_r --out pretrained/protoGDA_drug_cold
```

各场景配置与 `pretrained/` 目录的对应关系见 `THIRD_PARTY_NOTICES.md`。基线复现：

```bash
python scripts/cv_runner.py --model candela --config baselines/candela/config_drug_cold.yaml \
    --k 6 --split_by drug_cold --epochs 60 --patience 999 \
    --checkpoint-metric pearson_r --out pretrained/CANDELA_drug_cold
# --model mgataf 同理
```

### GDSC1 零样本外部验证（论文 5.3 节）

```bash
python scripts/external_validate_gdsc1.py --model cellquery   # 需 data/gdsc1.pkl
python scripts/external_validate_gdsc1.py --model candela
python scripts/external_validate_gdsc1.py --model mgataf
```

## 模型架构速览

```
Cell_Features ──► MLP Projector ──► Q (probes) ─────────────┐ (每层复用)
Drug_SMILES ──► Mol_Graph(+GMM) ──► EGNN/EQGAT 逐层演化 ──► CrossAttn(Q) ──► ...
   Morgan/ChemBERTa ──► 原子门控广播，进入各层消息传递            │
                                                                ▼
                                              M_accum ──► MLP ──► y_hat
```

## License 与第三方

- 主代码与 `pretrained/` 权重：**MIT License**（见 [LICENSE](LICENSE)）。
- 基线为论文的独立重写实现，第三方归属与出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
