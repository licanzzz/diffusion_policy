# Diffusion Policy 项目复现总结

## ✅ 已完成的设置

### 1. 系统依赖安装
已成功安装以下系统依赖：
- `libosmesa6-dev` - MuJoCo 渲染依赖
- `libglfw3` - OpenGL窗口管理
- `patchelf` - 二进制修补工具
- `mesa-utils` - Mesa 3D工具
- `libglu1-mesa-dev` - OpenGL实用程序库

### 2. Conda环境创建
- ✅ 已安装 Miniconda3 到 `~/miniconda3`
- ✅ 已创建名为 `robodiff` 的conda环境
- ✅ Python版本: 3.9.18
- ✅ 已安装所有主要依赖包：
  - PyTorch 1.12.1
  - CUDA Toolkit 11.6
  - NumPy 1.23.3
  - 其他所有在 `conda_environment.yaml` 中列出的包

### 3. 训练数据下载
- ✅ 已下载 Push-T 数据集
- ✅ 数据集位置: `/workspace/data/pusht/pusht_cchi_v7_replay.zarr`
- ✅ 数据集验证: 成功加载，包含以下数据字段：
  - `action`, `img`, `keypoint`, `n_contacts`, `state`

### 4. 配置文件
- ✅ 已有配置文件: `image_pusht_diffusion_policy_cnn.yaml`
- ✅ 配置文件路径: `/workspace/image_pusht_diffusion_policy_cnn.yaml`

### 5. diffusion_policy包安装
- ✅ 已将项目以可编辑模式安装到conda环境中

## ⚠️ 已知限制

### PyTorch执行栈问题
当前环境存在PyTorch库的执行栈限制问题：
```
ImportError: libtorch_cpu.so: cannot enable executable stack as shared object requires: Invalid argument
```

这是由于系统内核安全策略导致的。有以下几种解决方案：

#### 解决方案 1: 使用支持GPU的环境
这个项目设计用于GPU训练。建议在有NVIDIA GPU的环境中运行：
```bash
# 激活环境
conda activate robodiff

# 启动训练（需要GPU）
python train.py --config-dir=. --config-name=image_pusht_diffusion_policy_cnn.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

#### 解决方案 2: 在支持的Linux发行版上运行
推荐使用Ubuntu 20.04，这是项目官方测试的环境。

#### 解决方案 3: 使用Docker
可以使用Docker容器来避免系统级别的限制。

## 📝 下一步操作

### 基本使用
1. **激活conda环境**:
   ```bash
   conda activate robodiff
   ```

2. **登录wandb（用于训练日志）**:
   ```bash
   wandb login
   ```

3. **运行单次训练**:
   ```bash
   python train.py \
       --config-dir=. \
       --config-name=image_pusht_diffusion_policy_cnn.yaml \
       training.seed=42 \
       training.device=cuda:0 \
       hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
   ```

### 多种子训练
使用Ray进行并行训练：
```bash
# 启动ray集群
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3

# 运行多种子训练
python ray_train_multirun.py \
    --config-dir=. \
    --config-name=image_pusht_diffusion_policy_cnn.yaml \
    --seeds=42,43,44 \
    --monitor_key=test/mean_score \
    -- multi_run.run_dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
    multi_run.wandb_name_base='${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}'
```

### 评估预训练模型
```bash
# 下载检查点
wget -P data/ https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt

# 运行评估
python eval.py \
    --checkpoint data/epoch=0550-test_mean_score=0.969.ckpt \
    --output_dir data/pusht_eval_output \
    --device cuda:0
```

## 📚 项目资源

- **项目主页**: https://diffusion-policy.cs.columbia.edu/
- **论文**: https://diffusion-policy.cs.columbia.edu/#paper
- **训练数据**: https://diffusion-policy.cs.columbia.edu/data/training/
- **实验日志**: https://diffusion-policy.cs.columbia.edu/data/experiments/
- **Colab笔记本**:
  - 状态版本: https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B
  - 视觉版本: https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg

## 🔍 验证安装

### 验证数据集加载
```bash
$HOME/miniconda3/envs/robodiff/bin/python -c "
import zarr
store = zarr.open('/workspace/data/pusht/pusht_cchi_v7_replay.zarr', 'r')
print('Dataset loaded successfully')
print('Data keys:', list(store['data'].keys()))
print('Action shape:', store['data']['action'].shape)
print('Image shape:', store['data']['img'].shape)
"
```

### 验证基本依赖
```bash
$HOME/miniconda3/envs/robodiff/bin/python -c "
import numpy as np
import zarr
import hydra
print('✓ NumPy:', np.__version__)
print('✓ Zarr:', zarr.__version__)
print('✓ Hydra: installed')
print('All basic dependencies working!')
"
```

## 🛠️ 故障排除

### 如果遇到CUDA错误
项目需要NVIDIA GPU。如果没有GPU，某些功能将无法使用。

### 如果遇到依赖问题
重新创建conda环境：
```bash
conda env remove -n robodiff
conda env create -f conda_environment.yaml
```

### 如果数据集路径错误
确保数据集在正确位置：
```bash
ls -la /workspace/data/pusht/pusht_cchi_v7_replay.zarr
```

## 📧 获取帮助

- 查看项目README: `/workspace/README.md`
- 查看配置示例: `/workspace/diffusion_policy/config/`
- 查看测试用例: `/workspace/tests/`
