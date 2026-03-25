# Adversarial Robustness PyTorch

## Project Overview

非官方 PyTorch 实现，复现两篇对抗鲁棒性论文：
1. **Gowal et al., 2020** — "Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples"
2. **Rebuffi et al., 2021** — "Fixing Data Augmentation to Improve Adversarial Robustness"

支持 TRADES、MART、标准对抗训练，以及 EMA 权重平均、CutMix、半监督学习等技术。

## Project Structure

```
├── train.py                           # 标准对抗训练入口
├── train-wa.py                        # 带权重平均 (EMA) 的训练 (Gowal 方法)
├── eval-aa.py                         # AutoAttack 评估
├── eval-adv.py                        # PGD 白盒/黑盒评估
├── eval-rb.py                         # RobustBench 评估
├── run_cifar10_cutmix.slurm           # SLURM 任务示例
├── core/
│   ├── models/                        # 模型架构
│   │   ├── wideresnet.py              #   WideResNet (标准)
│   │   ├── wideresnetwithswish.py     #   WideResNet + Swish (Gowal 版)
│   │   ├── resnet.py                  #   标准 ResNet
│   │   ├── preact_resnet.py           #   PreAct ResNet
│   │   └── preact_resnetwithswish.py  #   PreAct ResNet + Swish
│   ├── attacks/                       # 攻击算法
│   │   ├── pgd.py                     #   PGD (L∞/L2)
│   │   ├── fgsm.py                    #   FGSM/FGM
│   │   ├── apgd.py                    #   APGD (AutoAttack 变体)
│   │   └── deepfool.py                #   DeepFool
│   ├── data/                          # 数据加载
│   │   ├── cifar10.py, cifar100.py    #   标准数据集
│   │   ├── cifar10s.py, cifar100s.py  #   半监督数据集
│   │   └── semisup.py                 #   半监督采样工具
│   └── utils/
│       ├── train.py                   #   Trainer 类 (核心训练逻辑)
│       ├── trades.py                  #   TRADES loss
│       ├── mart.py                    #   MART loss
│       ├── parser.py                  #   参数解析
│       └── utils.py                   #   SmoothCrossEntropyLoss 等工具
└── gowal21uncovering/utils/
    ├── watrain.py                     # WATrainer (EMA 权重平均)
    ├── trades.py                      # 增强 TRADES (label smoothing)
    └── cutmix.py                      # CutMix 数据增强
```

## Dependencies

- Python 3.x, PyTorch 1.8+, torchvision
- autoattack, robustbench, numpy, pandas, scipy, tqdm

## Supported Models

通过 `--model` 参数指定：
- WideResNet: `wrn-28-10`, `wrn-34-10`, `wrn-34-20`, `wrn-70-16`
- WideResNet + Swish: `wrn-28-10-swish`, `wrn-34-10-swish`, `wrn-34-20-swish`, `wrn-70-16-swish`
- ResNet: `resnet18`, `resnet34`, `resnet50`
- PreAct ResNet: `preact-resnet18/34/50/101` (标准 + Swish 变体)

Swish 版本 (Gowal) 内置 mean/std 归一化层，标准版本不内置。

## Key Hyperparameters

| 参数 | CIFAR-10/100 | 说明 |
|------|-------------|------|
| epsilon | 8/255 | 扰动预算 |
| attack-step (训练) | 2/255 | PGD 步长 |
| attack-step (评估) | 1/255 | 评估 PGD 步长 |
| attack-iter (训练) | 10 | PGD 迭代次数 |
| attack-iter (评估) | 20-40 | 评估迭代次数 |
| beta (TRADES) | 6.0 | 鲁棒性权重 |
| lr | 0.1~0.4 | 学习率 |
| weight-decay | 5e-4 | 权重衰减 |
| tau (EMA) | 0.999 | 权重平均衰减率 |
| batch-size | 128~1024 | 批大小 |
| scheduler | cosinew | OneCycleLR (推荐) |
| CutMix size | 20 | 固定窗口大小 (CIFAR 32×32) |

## Common Commands

### 训练

```bash
# 标准 TRADES 训练
python train.py \
  --data-dir ./data --log-dir ./logs \
  --desc wrn34-trades --data cifar10 \
  --model wrn-34-10 --num-adv-epochs 200 \
  --beta 6.0 --lr 0.1

# 带权重平均 + CutMix 训练 (Gowal/Rebuffi 方法)
python train-wa.py \
  --data-dir ~/data/cifar10s --log-dir ~/logs \
  --desc wrn34-10-cutmix --data cifar10s \
  --batch-size 512 --model wrn-34-10-swish \
  --num-adv-epochs 400 --lr 0.2 --beta 6.0 \
  --unsup-fraction 0.0 --cutmix
```

### 评估

```bash
# AutoAttack (标准评估)
python eval-aa.py --data-dir ./data --log-dir ./logs \
  --desc wrn34-trades --data cifar10 --model wrn-34-10

# PGD 白盒评估
python eval-adv.py --data-dir ./data --log-dir ./logs \
  --desc wrn34-trades --data cifar10 --model wrn-34-10

# RobustBench 评估
python eval-rb.py --data-dir ./data --log-dir ./logs \
  --desc wrn34-trades --data cifar10 --model wrn-34-10
```

## 两个训练入口的区别

| 特性 | train.py | train-wa.py |
|------|----------|-------------|
| EMA 权重平均 | 无 | 有 (WATrainer) |
| CutMix | 不支持 | 支持 (`--cutmix`) |
| 半监督数据 | 不支持 | 支持 (cifar10s/100s) |
| 自动 resume | 无 | 有 (检测 weights-last.pt) |
| BN momentum 处理 | 标准 | 特殊处理 (0.01) |
| 适用场景 | 快速实验 | 复现 SOTA 结果 |

## Loss 函数

### TRADES (core/utils/trades.py)
```
L = CE(f(x), y) + β * KL(f(x) || f(x'))
```
内循环 PGD 最大化 KL 散度生成对抗样本。

### MART (core/utils/mart.py)
```
L = BCE_margin(f(x'), y) + β * weighted_KL(f(x) || f(x'))
```
使用 margin-based loss，权重基于置信度。通过 `--mart` 启用。

### 增强 TRADES (gowal21uncovering/utils/trades.py)
在标准 TRADES 基础上支持 label smoothing 和 soft labels (CutMix)。

## Checkpoint Formats

```python
# train.py (简单格式)
{'model_state_dict': model.state_dict()}

# train-wa.py (完整格式，支持 resume)
{
    'model_state_dict': wa_model.state_dict(),         # EMA 平均权重
    'unaveraged_model_state_dict': model.state_dict(), # 原始权重
    'epoch': epoch,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'old_score': [clean_acc, adv_acc]
}
```

## 输出目录结构

```
logs/<desc>/
├── args.txt              # 超参数 JSON
├── log-train.log         # 训练日志
├── stats_adv.csv         # 指标表 (loss, acc, lr)
├── weights-best.pt       # 最佳 checkpoint (按 adv acc)
├── weights-last.pt       # 最新 checkpoint (resume 用)
├── log-aa.log            # AutoAttack 结果
└── log-adv.log           # PGD 评估结果
```

## Coding Conventions

- 数据输入 raw [0,1]，归一化在模型内部完成 (Normalization 层)
- epsilon/step_size 用分数传入 (如 `--attack-eps 8 --attack-step 2`，内部 /255)
- 评估用 2× 训练迭代次数的 PGD
- Swish 模型 (`*-swish`) 使用内置归一化，标准模型使用外部 Normalization wrapper
- 半监督数据集名带 `s` 后缀 (cifar10s, cifar100s)
