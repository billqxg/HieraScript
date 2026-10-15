# HieraScript / PRCV2026

HieraScript 是一个面向多语言场景文字图像的三级层次识别实现。模型同时预测：

1. Macro writing system（宏观书写类别）；
2. Script（文字体系）；
3. Language（语言类别）。

当前 HSI-3L 配置包含 5 个宏观类别、19 个文字体系、33 个语言类别。代码以 CLIP 视觉编码器为基础，通过层级专用分支、渐进式特征融合、层次一致性约束和可选的原型置信度校准进行训练与评估。

## 1. 目录结构

```text
PRCV2026/
├── clip/                       # 随仓库保存的 CLIP 实现与 tokenizer 资源
├── confidence_calibrator.py    # 基于类别原型和特征距离的置信度校准
├── dataset_multilang.py        # 三级目录数据集读取与类别树构建
├── model_hrn_clip.py           # ResNet/ViT 版层次识别网络
├── train_hrn.py                # 训练、评估、日志和 checkpoint 入口
├── utils_multilang.py          # 层次映射、指标和辅助损失
├── requirements.txt
├── run.sh                      # RN50 默认训练示例
├── data -> ...                 # HSI-3L 数据软链接，不纳入 Git
├── pretrained -> ...           # CLIP 权重软链接，不纳入 Git
└── results/                    # 日志和模型输出，不纳入 Git
```

服务器上的本版仓库位于：

```text
/home/mtr/PRCV2026
```

## 2. 数据目录约定

`MultilangDataset` 从三级目录名自动建立类别集合和父子映射。训练集与测试集必须具有一致的层次结构：

```text
data/
├── train/
│   └── <macro>/<script>/<language>/*.{jpg,jpeg,png,bmp,webp,tif,tiff,gif}
└── test/
    └── <macro>/<script>/<language>/*.{jpg,jpeg,png,bmp,webp,tif,tiff,gif}
```

例如：

```text
data/train/Alphabetic_Linear/Latin/English/example.jpg
data/test/Abjad_Cursive/Arabic/Persian/example.png
```

类别 ID 按目录名称排序后生成。请勿在训练集和测试集中使用不同的目录命名，否则会造成标签映射不一致。数据加载阶段会忽略非目录项和不支持的文件扩展名。

当前服务器通过软链接使用原始数据，不复制约 47 万幅图像：

```bash
ln -s /home/ll/PRCV2026/HSI-3L-dataset data
```

迁移到其他机器时，只需重新建立 `data` 链接，或在命令行使用 `--data_path /path/to/HSI-3L-dataset`。

## 3. 环境安装

建议使用 Python 3.8，与当前固定的 PyTorch 1.13.1 依赖匹配。

```bash
cd /home/mtr/PRCV2026
conda create -n hierascript python=3.8 -y
conda activate hierascript
pip install -r requirements.txt
```

主要依赖：

- PyTorch 1.13.1 / torchvision 0.14.1
- NumPy 1.21.6
- Pillow 9.5.0
- NetworkX 2.8.8
- ftfy、regex、tqdm

`requirements.txt` 固定了 Python 包版本，但 CUDA 与 NVIDIA 驱动需根据服务器环境单独确认。可先执行：

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda devices:', torch.cuda.device_count())
PY
```

## 4. 预训练权重

模型需要与 `--backbone` 对应的 CLIP 视觉预训练权重。服务器当前建立：

```text
pretrained -> /home/ll/HRN_FINAL
```

默认会读取 `pretrained/<BACKBONE>.pt`。也可以显式指定：

```bash
--clip_pretrained /path/to/RN50.pt
```

支持的 `--backbone`：

- `RN50`
- `RN50x4`
- `RN50x16`
- `RN50x64`
- `ViT-B-16`
- `ViT-B-32`
- `ViT-L-14`

代码会从权重自动推断视觉编码器结构。不同骨干使用不同输入尺寸：RN50 为 224，RN50x4 为 288，RN50x16 为 384，RN50x64 为 448；其余默认使用 224。

## 5. 快速训练

默认 RN50 实验：

```bash
bash run.sh
```

等价的显式命令：

```bash
python train_hrn.py \
  --backbone RN50 \
  --clip_pretrained pretrained/RN50.pt \
  --data_path data \
  --save_dir results \
  --freeze_clip \
  --epoch 20 \
  --batch 32 \
  --seed 42 \
  --device 0 \
  --lr_adjt Cos
```

`run.sh` 可通过环境变量覆盖常用配置：

```bash
DEVICE=1 BACKBONE=RN50x16 BATCH=16 EPOCHS=20 bash run.sh
```

后台运行示例：

```bash
nohup env DEVICE=0 BACKBONE=RN50 bash run.sh > results/launcher.log 2>&1 &
```

## 6. 方法开关

基础训练始终包含三级交叉熵、从细粒度到粗粒度的概率一致性约束，以及层次置信度违反惩罚。以下参数控制额外实验配置。

### 6.1 冻结 CLIP 视觉编码器

```bash
--freeze_clip
```

只训练 HRN 分支、投影层、三级分类头和可学习损失权重。若不传该参数，视觉骨干也会参与优化，并使用更低学习率。

### 6.2 自适应三级交叉熵权重

```bash
--use_adaptive_ce --adaptive_gamma 0.5
```

根据 Macro、Script、Language 三项当前损失动态分配权重。`adaptive_gamma` 越小，权重分布越集中。

### 6.3 TreeLoss 辅助约束

```bash
--use_tree --tree_weight 0.1
```

在主损失之外加入类别树辅助正则项。

### 6.4 原型置信度校准

```bash
--use_conf_calib \
--conf_beta 10.0 \
--conf_threshold 0.5 \
--calib_start_epoch 0
```

校准器使用训练集中正确分类样本的层级特征均值构造类别原型，再用测试特征到原型的余弦距离修正 softmax 概率。`conf_threshold > 0` 时，还会报告高于阈值样本上的准确率、H-F1 和不可靠样本比例。

注意：当前实现会在评估前重新遍历训练集拟合原型。对完整 HSI-3L 使用该选项会增加每轮评估时间。

## 7. 仅评估 checkpoint

```bash
python train_hrn.py \
  --backbone RN50 \
  --clip_pretrained pretrained/RN50.pt \
  --data_path data \
  --save_dir results \
  --model results/best_hrn4clip_RN50_e20_b32_frozen.pth \
  --freeze_clip \
  --eval_only \
  --device 0
```

如需在评估时拟合并启用置信度校准，在上述命令中增加 `--use_conf_calib`。checkpoint 中保存 `epoch`、`state_dict` 和优化器状态；`--model` 加载网络参数，但当前入口不会从保存的 epoch 自动继续训练计数。

## 8. 训练输出

`--save_dir` 下主要产生：

```text
results/
├── best_hrn4clip_<backbone>_e<epochs>_b<batch>[_frozen...].pth
├── logs/
│   └── hrn4clip_<backbone>_e<epochs>_b<batch>[_frozen...].json
└── run.log
```

- `best_*.pth`：按测试集 H-F1 保存的最佳 checkpoint；
- `logs/*.json`：逐 epoch 的训练/测试指标、耗时和最佳 epoch；
- `run.log`：`run.sh` 的标准输出和错误输出。

主要指标包括：

- `macro_acc`：宏观书写类别准确率；
- `script_acc`：文字体系准确率；
- `language_acc`：语言准确率；
- `hierarchical_f1`：预测路径与真实路径的层次 F1；
- `severity_of_mistake`：层次错误严重度；
- 启用置信度阈值后，还包括 `confident_accuracy`、`confident_h_f1` 和 `unreliable_rate`。

## 9. 常用参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--worker` | 4 | DataLoader 进程数 |
| `--epoch` | 20 | 训练轮数 |
| `--batch` | 32 | 批大小 |
| `--seed` | 42 | PyTorch 随机种子 |
| `--device` | `0` | GPU 编号；可传多个编号启用 DataParallel |
| `--lr_adjt` | `Cos` | `Cos` 或 `Step` 学习率策略 |
| `--data_path` | `data` | 包含 `train/` 和 `test/` 的数据根目录 |
| `--save_dir` | `results` | checkpoint 与日志目录 |
| `--model` | 空 | 待加载的训练 checkpoint |
| `--clip_pretrained` | 自动 | CLIP 视觉权重路径 |

多 GPU 示例：

```bash
python train_hrn.py --device 0 1 --data_path data --freeze_clip
```

## 10. 复现实验建议

1. 确认 `data/train` 与 `data/test` 的类别目录完全一致；
2. 记录 Git commit、数据软链接目标和预训练权重文件；
3. 使用固定的 `--seed 42`，并同时记录 GPU、CUDA、PyTorch 版本；
4. 先用较小 batch 和少量 epoch 做连通性测试，再启动完整训练；
5. 跨数据集测试时不要修改目标数据标签映射，也不要把目标测试集用于训练；
6. 比较消融实验时保持 backbone、数据划分、seed、batch 和训练轮数一致。

快速连通性测试：

```bash
python train_hrn.py \
  --backbone RN50 \
  --data_path data \
  --save_dir results/smoke_test \
  --freeze_clip \
  --epoch 1 \
  --batch 8 \
  --worker 2 \
  --device 0
```

## 11. 常见问题

### 找不到数据或类别数为零

检查软链接和目录层级：

```bash
readlink -f data
find data/train -mindepth 3 -maxdepth 3 -type d | head
find data/test  -mindepth 3 -maxdepth 3 -type d | head
```

### 找不到 CLIP 权重

检查 `pretrained/<BACKBONE>.pt`，或显式传入 `--clip_pretrained`。权重文件名必须与选择的模型结构匹配。

### CUDA out of memory

降低 `--batch`；RN50x16、RN50x64 和较大的 ViT 通常需要更小 batch。多 GPU 的 `DataParallel` 不会降低主卡保存输出和聚合操作的全部开销。

### DataLoader worker 报错

先将 `--worker 0` 验证是否为多进程读取问题，同时检查损坏图像和目录访问权限。

### 评估速度明显变慢

启用 `--use_conf_calib` 后，校准器需要额外遍历训练集计算类别原型；这是当前实现的预期行为。

## 12. Git 与大文件约定

数据集、预训练权重、训练日志和 checkpoint 均不进入 Git。仓库保留 `data` 与 `pretrained` 的相对入口约定，但实际软链接由每台机器自行建立。提交代码前建议执行：

```bash
git status --short
git ls-files | grep -E '\.(pth|pt|ckpt)$' && echo '请移除模型大文件'
```

本仓库用于项目结题归档与研究复现。公开发布前仍需确认各数据源和 CLIP 权重的许可证及再分发条件。
