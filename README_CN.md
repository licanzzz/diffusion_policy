# Diffusion Policy - 中文使用指南

这是 Diffusion Policy 项目的中文使用指南。项目已经完成基本设置，您可以开始使用了！

## 🎉 项目设置状态

✅ **所有基础组件已成功安装和配置**

### 已完成的设置包括：

1. ✅ 系统依赖（MuJoCo相关库）
2. ✅ Conda环境 `robodiff` (Python 3.9.18)
3. ✅ 所有Python依赖包（PyTorch, NumPy, Zarr等）
4. ✅ Push-T训练数据集下载
5. ✅ 配置文件准备
6. ✅ diffusion_policy包安装

## 🚀 快速开始

### 方法一：使用快速启动脚本（推荐）

```bash
./quick_start.sh
```

脚本提供以下功能：
1. 验证安装
2. 查看数据集信息
3. 显示训练命令示例
4. 显示评估命令示例
5. 打开Python交互式环境

### 方法二：手动验证

```bash
# 验证基本依赖
$HOME/miniconda3/envs/robodiff/bin/python -c "
import numpy as np
import zarr
import hydra
import diffusion_policy
print('✓ 所有依赖正常工作！')
"

# 查看数据集
$HOME/miniconda3/envs/robodiff/bin/python -c "
import zarr
store = zarr.open('data/pusht/pusht_cchi_v7_replay.zarr', 'r')
print('数据集字段:', list(store['data'].keys()))
print('动作数据形状:', store['data']['action'].shape)
"
```

## 📖 详细文档

请查看 `PROJECT_SETUP_SUMMARY.md` 获取：
- 完整的安装详情
- 已知问题和解决方案
- 训练和评估的详细命令
- 故障排除指南

## 🎯 下一步操作

### 如果您有GPU环境

1. **激活conda环境**:
   ```bash
   conda activate robodiff
   ```

2. **配置wandb（可选，用于训练日志）**:
   ```bash
   wandb login
   ```

3. **开始训练**:
   ```bash
   python train.py \
       --config-dir=. \
       --config-name=image_pusht_diffusion_policy_cnn.yaml \
       training.seed=42 \
       training.device=cuda:0 \
       hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
   ```

### 如果没有GPU环境

此项目主要设计用于GPU训练。建议：

1. **使用Google Colab**（免费GPU）:
   - [状态版本笔记本](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B)
   - [视觉版本笔记本](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg)

2. **探索代码和数据**:
   ```bash
   # 查看项目结构
   tree -L 2 diffusion_policy/
   
   # 查看配置文件
   cat diffusion_policy/config/task/pusht_image.yaml
   
   # 探索数据集
   ./quick_start.sh  # 选择选项2查看数据集信息
   ```

3. **阅读论文和文档**:
   - [项目主页](https://diffusion-policy.cs.columbia.edu/)
   - [论文](https://diffusion-policy.cs.columbia.edu/#paper)
   - 本地README: `README.md`

## 📁 项目结构

```
/workspace/
├── README.md                                    # 原始英文文档
├── README_CN.md                                 # 本文档（中文）
├── PROJECT_SETUP_SUMMARY.md                     # 详细设置总结
├── quick_start.sh                               # 快速启动脚本
├── train.py                                     # 训练入口
├── eval.py                                      # 评估入口
├── image_pusht_diffusion_policy_cnn.yaml       # 配置文件
├── conda_environment.yaml                       # Conda环境配置
├── data/
│   └── pusht/                                   # Push-T数据集
│       └── pusht_cchi_v7_replay.zarr/
├── diffusion_policy/                            # 核心代码
│   ├── config/                                  # 配置文件
│   ├── dataset/                                 # 数据集实现
│   ├── env/                                     # 环境实现
│   ├── model/                                   # 模型实现
│   ├── policy/                                  # 策略实现
│   └── workspace/                               # 工作空间实现
└── tests/                                       # 测试代码
```

## 🔧 常见问题

### Q: 如何检查我的环境是否正确设置？

A: 运行快速启动脚本并选择选项1：
```bash
./quick_start.sh  # 然后输入 1
```

### Q: 数据集存放在哪里？

A: `/workspace/data/pusht/pusht_cchi_v7_replay.zarr`

查看数据集信息：
```bash
./quick_start.sh  # 然后输入 2
```

### Q: 我没有GPU可以训练吗？

A: 这个项目需要GPU进行训练。建议使用：
- Google Colab（免费GPU）
- 云服务提供商（AWS, GCP, Azure等）
- 本地GPU工作站

### Q: 如何下载其他数据集？

A: 访问 [训练数据页面](https://diffusion-policy.cs.columbia.edu/data/training/)，然后：
```bash
cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/<dataset_name>.zip
unzip <dataset_name>.zip
```

### Q: 在哪里可以找到预训练模型？

A: 访问 [实验日志页面](https://diffusion-policy.cs.columbia.edu/data/experiments/)

下载示例：
```bash
mkdir -p data/checkpoints
wget -P data/checkpoints/ \
    https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt
```

## 📚 学习资源

1. **官方资源**:
   - [项目主页](https://diffusion-policy.cs.columbia.edu/)
   - [论文PDF](https://diffusion-policy.cs.columbia.edu/#paper)
   - [实验日志和检查点](https://diffusion-policy.cs.columbia.edu/data/experiments/)

2. **交互式教程**:
   - [Google Colab - 状态版本](https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B)
   - [Google Colab - 视觉版本](https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg)

3. **代码导航**:
   - 任务实现: `diffusion_policy/dataset/` 和 `diffusion_policy/env_runner/`
   - 策略实现: `diffusion_policy/policy/`
   - 模型实现: `diffusion_policy/model/`
   - 配置文件: `diffusion_policy/config/`

## 💡 提示和最佳实践

1. **首次使用建议先在Colab上实验**，理解工作流程后再在本地训练

2. **使用wandb追踪实验**:
   ```bash
   wandb login  # 首次使用时运行
   ```

3. **多种子训练以获得稳定结果**（如果有足够的GPU资源）

4. **查看配置文件**了解可调整的超参数:
   ```bash
   cat image_pusht_diffusion_policy_cnn.yaml
   ```

5. **参考tests/目录**中的测试用例了解各组件的使用方法

## 🆘 获取帮助

如果遇到问题：

1. 查看 `PROJECT_SETUP_SUMMARY.md` 的故障排除部分
2. 运行 `./quick_start.sh` 验证安装
3. 查看项目的GitHub Issues
4. 阅读原始README.md中的详细文档

## ✨ 开始探索

现在环境已经准备就绪，祝您使用愉快！建议从以下步骤开始：

1. ✅ 运行 `./quick_start.sh` 验证安装
2. 📖 阅读论文理解方法
3. 🎮 在Colab上运行交互式笔记本
4. 💻 如果有GPU，尝试本地训练
5. 🔬 探索不同的任务和配置

Happy coding! 🚀
