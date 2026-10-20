# Airport Cargo MAWM

面向机场地面协同货运的决策校准多智能体世界模型规划（Multi-Agent World Model, MAWM）研究代码。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/june250630/airport-cargo-mawm/blob/main/notebooks/00_colab_quickstart.ipynb)

## 当前状态

仓库目前提供一个**独立、最小、可运行的状态空间基线**：

- `AirportCargoEnv`：PettingZoo `ParallelEnv` 双智能体机场货运环境；
- 两辆牵引车只有在同时靠近、同向施力时才能移动 ULD（Unit Load Device，航空集装器）；
- 脚本专家：用于任务可解性检查和训练数据生成，不作为学习方法基线；
- HDF5 数据采集：按完整场景记录状态、联合动作和下一状态；
- `StateWorldModel`：从全局状态和联合动作预测下一状态；
- CEM（Cross-Entropy Method，交叉熵方法）规划；
- 多模型集成分歧作为不确定性代价，为后续“决策校准”实验提供接口；
- 完整训练断点包含模型、优化器、epoch、场景划分和历史指标，适合 Colab 断线恢复。

这不是像素 JEPA 的最终实现。当前版本先把“环境 → 专家 → 数据 → 模型 → 规划 → 评估”闭环跑通；VMAS 场景和像素 LeWM/JEPA 是下一阶段。

## 快速开始

核心MVP支持Python 3.10–3.13；后续接入`stable-worldmodel`时以其版本约束为准：

```bash
python -m pip install -e ".[dev]"
pytest
```

### 1. 验证环境与脚本专家

```bash
python -m airport_cargo_mawm.cli.smoke --episodes 5 --videos 2
```

输出位于 `outputs/smoke/`，包括 `results.json` 和 MP4 视频。

### 2. 采集小规模数据

```bash
python -m airport_cargo_mawm.cli.collect \
  --episodes 100 \
  --seed 0 \
  --output data/airport_cargo_train.h5
```

另用不同随机种子采集保留测试集：

```bash
python -m airport_cargo_mawm.cli.collect \
  --episodes 20 \
  --seed 10000 \
  --output data/airport_cargo_heldout.h5
```

### 3. 训练状态世界模型

```bash
python -m airport_cargo_mawm.cli.train \
  --data data/airport_cargo_train.h5 \
  --output checkpoints/state_wm_seed0 \
  --epochs 30 \
  --seed 0
```

从断点继续：

```bash
python -m airport_cargo_mawm.cli.train \
  --data data/airport_cargo_train.h5 \
  --output checkpoints/state_wm_seed0 \
  --epochs 60 \
  --seed 0 \
  --resume checkpoints/state_wm_seed0/last.pt
```

`--epochs 60` 表示训练到总计 60 个 epoch，而不是再运行 60 个。

### 4. 用学习到的模型做 CEM 规划

单模型：

```bash
python -m airport_cargo_mawm.cli.evaluate \
  --checkpoint checkpoints/state_wm_seed0/last.pt \
  --episodes 10 \
  --output outputs/cem_single
```

训练多个随机种子后，可以启用集成不确定性惩罚：

```bash
python -m airport_cargo_mawm.cli.evaluate \
  --checkpoint \
    checkpoints/state_wm_seed0/last.pt \
    checkpoints/state_wm_seed1/last.pt \
    checkpoints/state_wm_seed2/last.pt \
  --uncertainty-weight 10.0 \
  --episodes 10 \
  --output outputs/cem_calibrated
```

## 状态和动作定义

全局状态共 11 维：

```text
[tug0_x, tug0_y,
 tug1_x, tug1_y,
 cargo_x, cargo_y,
 cargo_vx, cargo_vy,
 goal_x, goal_y,
 time_fraction]
```

联合动作共 4 维：

```text
[tug0_ax, tug0_ay, tug1_ax, tug1_ay]
```

位置和动作采用归一化坐标，动作范围为 `[-1, 1]`。

## 实验方法学

推荐依次建立四个层次：

1. 脚本专家：验证任务可解并生成数据；
2. 真实仿真器 + 真实代价 + CEM：测量规划搜索上限；
3. 状态世界模型 + CEM：分离动力学学习误差与视觉误差；
4. 像素 JEPA 世界模型 + 校准 CEM：最终研究方法。

训练集和验证集必须按完整 `scene_id` 划分，不能随机拆分相邻帧。本仓库的训练入口已按场景划分。

## Colab 数据管理

- GitHub保存代码；
- `/content`保存当前运行时的活动数据，读写更快；
- Google Drive保存HDF5、checkpoint、日志和最终结果；
- 免费Colab可能断开，正式训练必须逐epoch保存并测试恢复功能。

请从 `notebooks/00_colab_quickstart.ipynb` 开始。

## 后续路线

- [ ] 实现真实仿真器 CEM 上限基线；
- [ ] 增加 VMAS `AirportCargoScenario` 和 PettingZoo 适配；
- [ ] 增加像素数据字段与图像编码器；
- [ ] 独立实现 LeWM 风格 JEPA：预测损失 + 表征正则；
- [ ] 加入质量、摩擦、障碍、通信延迟和智能体故障等分布外条件；
- [ ] 比较普通 CEM、模型集成校准 CEM 和约束规划；
- [ ] 固化 AICAP 2027 实验协议、随机种子和统计报告。

## 来源与许可

本项目采用 Apache-2.0 许可证。研究设计参考并应引用：

- LeWorldModel（LeWM）论文及其官方实现；
- `stable-worldmodel` 的可复现数据、规划与评估接口；
- PettingZoo 多智能体环境接口；
- VMAS 向量化多智能体仿真器。

本仓库不包含或声称拥有学长未公开的模型权重。
