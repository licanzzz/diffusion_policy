# 训练进程资源占用问题分析报告

## 🚨 问题描述

您运行训练脚本后出现了大量Python子进程，导致系统资源耗尽的问题。

## 📊 问题分析

### 发现的问题：
1. **进程爆炸**: 一个训练任务创建了60多个Python子进程
2. **内存占用**: 每个进程占用约800MB内存，总共约48GB内存需求
3. **系统资源**: 超出了15GB的系统内存限制

### 根本原因：
这是由于Diffusion Policy的**多进程数据加载器**和**环境并行化**配置不当导致的：

1. **DataLoader workers**: `dataloader.num_workers=1` 但可能有其他并行设置
2. **Environment parallelization**: 环境评估时可能启用了大量并行环境
3. **Ray/多进程**: 可能启用了Ray集群或其他多进程框架

## 🔧 解决方案

### 1. 修改数据加载器配置
```yaml
dataloader:
  batch_size: 4
  num_workers: 0  # 禁用多进程数据加载
  shuffle: True
  pin_memory: False  # 在CPU模式下禁用
  persistent_workers: False
```

### 2. 修改环境配置
```yaml
env_runner:
  n_envs: 1  # 限制并行环境数量
  n_train: 2  # 减少训练环境数量
  n_test: 5   # 减少测试环境数量
```

### 3. 禁用Ray集群
确保没有启用Ray多进程：
```bash
# 检查是否有Ray进程
ps aux | grep ray
# 如果有，停止Ray
ray stop
```

## 🛠️ 推荐的训练配置

创建一个资源友好的配置文件：

```bash
# 使用以下参数运行训练
cd /workspace
source $HOME/miniconda/bin/activate robodiff

python train.py \
  --config-dir=. \
  --config-name=train_diffusion_unet_lowdim_workspace \
  task=pusht_lowdim \
  training.device=cpu \
  training.num_epochs=5 \
  training.max_train_steps=100 \
  training.rollout_every=50 \
  training.checkpoint_every=50 \
  dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  logging.mode=disabled \
  env_runner.n_envs=1 \
  env_runner.n_train=2 \
  env_runner.n_test=5
```

## 📈 监控建议

### 训练前检查：
```bash
# 检查内存使用
free -h

# 检查磁盘空间
df -h

# 检查现有Python进程
ps aux | grep python | wc -l
```

### 训练中监控：
```bash
# 监控进程数量
watch "ps aux | grep python | wc -l"

# 监控内存使用
watch "free -h"

# 监控训练日志
tail -f data/outputs/*/train.log
```

### 紧急停止：
```bash
# 如果再次出现进程爆炸，立即执行：
pkill -9 -f "train.py"
```

## 🎯 优化建议

### 1. 使用GPU（如果可用）
```bash
# 安装CUDA版本的PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 使用GPU训练
training.device=cuda:0
```

### 2. 调整批次大小
- CPU训练：`batch_size=2-4`
- GPU训练：`batch_size=16-32`

### 3. 减少模型复杂度
```yaml
model:
  down_dims: [128, 256, 512]  # 减少模型大小
  diffusion_step_embed_dim: 128
```

## 🔍 故障排除

### 如果训练卡住：
1. 检查进程数量：`ps aux | grep python | wc -l`
2. 检查内存使用：`free -h`
3. 检查日志文件：`tail -f data/outputs/*/train.log`

### 如果内存不足：
1. 减少批次大小：`dataloader.batch_size=1`
2. 禁用数据预加载：`dataloader.pin_memory=False`
3. 减少并行环境：`env_runner.n_envs=1`

### 如果磁盘空间不足：
1. 清理缓存：`rm -rf ~/.cache/pip/*`
2. 清理临时文件：`rm -rf /tmp/*`
3. 清理旧的训练输出：`rm -rf data/outputs/*/`

## ✅ 验证步骤

运行修复后的训练：
1. 确保所有旧进程已清理
2. 使用推荐的配置参数
3. 监控进程数量和内存使用
4. 检查训练日志是否正常

## 📝 总结

这个问题是由于多进程配置不当导致的资源耗尽。通过：
1. 禁用多进程数据加载
2. 限制并行环境数量
3. 使用合适的批次大小
4. 监控系统资源

可以避免类似问题的再次发生。