#!/bin/bash

# Diffusion Policy 快速启动脚本
# Quick Start Script for Diffusion Policy

echo "======================================"
echo "Diffusion Policy 项目快速启动"
echo "======================================"
echo ""

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查conda是否已安装
if [ ! -d "$HOME/miniconda3" ]; then
    echo -e "${RED}错误: Miniconda未安装在 $HOME/miniconda3${NC}"
    echo "请先运行项目设置脚本"
    exit 1
fi

# 检查robodiff环境是否存在
if [ ! -d "$HOME/miniconda3/envs/robodiff" ]; then
    echo -e "${RED}错误: robodiff conda环境不存在${NC}"
    echo "请先运行: conda env create -f conda_environment.yaml"
    exit 1
fi

# 检查数据集是否存在
if [ ! -d "/workspace/data/pusht" ]; then
    echo -e "${YELLOW}警告: 训练数据不存在，正在下载...${NC}"
    mkdir -p /workspace/data
    cd /workspace/data
    wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
    unzip pusht.zip
    rm pusht.zip
    cd /workspace
fi

echo -e "${GREEN}✓ 环境检查完成${NC}"
echo ""

# 显示菜单
echo "请选择操作："
echo "1) 验证安装（推荐首次使用）"
echo "2) 查看数据集信息"
echo "3) 显示训练命令示例"
echo "4) 显示评估命令示例"
echo "5) 打开Python交互式环境"
echo "6) 退出"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo -e "${GREEN}正在验证安装...${NC}"
        echo ""
        echo "检查Python版本："
        $HOME/miniconda3/envs/robodiff/bin/python --version
        echo ""
        echo "检查基本依赖："
        $HOME/miniconda3/envs/robodiff/bin/python -c "
import numpy as np
import zarr
import hydra
import diffusion_policy
print('✓ NumPy:', np.__version__)
print('✓ Zarr:', zarr.__version__)
print('✓ Hydra: OK')
print('✓ Diffusion Policy: OK')
"
        echo ""
        echo "检查数据集："
        $HOME/miniconda3/envs/robodiff/bin/python -c "
import zarr
store = zarr.open('/workspace/data/pusht/pusht_cchi_v7_replay.zarr', 'r')
print('✓ 数据集加载成功')
print('  数据键:', list(store['data'].keys()))
print('  动作形状:', store['data']['action'].shape)
print('  图像形状:', store['data']['img'].shape)
"
        echo ""
        echo -e "${GREEN}✓ 验证完成！所有基本组件工作正常${NC}"
        ;;
    
    2)
        echo -e "${GREEN}数据集信息：${NC}"
        $HOME/miniconda3/envs/robodiff/bin/python -c "
import zarr
import numpy as np

store = zarr.open('/workspace/data/pusht/pusht_cchi_v7_replay.zarr', 'r')
print('数据集路径: /workspace/data/pusht/pusht_cchi_v7_replay.zarr')
print('')
print('数据字段:')
for key in store['data'].keys():
    data = store['data'][key]
    print(f'  {key:12s}: shape={data.shape}, dtype={data.dtype}')
print('')
print('元数据:')
for key in store['meta'].keys():
    meta = store['meta'][key]
    print(f'  {key:15s}: shape={meta.shape}, dtype={meta.dtype}')
    if key == 'episode_ends':
        print(f'    总episode数: {len(meta)}')
print('')
"
        ;;
    
    3)
        echo -e "${GREEN}训练命令示例：${NC}"
        echo ""
        echo "1. 激活conda环境："
        echo "   conda activate robodiff"
        echo ""
        echo "2. 登录wandb（首次使用需要）："
        echo "   wandb login"
        echo ""
        echo "3. 单次训练（需要GPU）："
        echo "   python train.py \\"
        echo "       --config-dir=. \\"
        echo "       --config-name=image_pusht_diffusion_policy_cnn.yaml \\"
        echo "       training.seed=42 \\"
        echo "       training.device=cuda:0 \\"
        echo "       hydra.run.dir='data/outputs/\${now:%Y.%m.%d}/\${now:%H.%M.%S}_\${name}_\${task_name}'"
        echo ""
        echo "4. 多种子训练（需要GPU和Ray）："
        echo "   # 首先启动ray集群"
        echo "   export CUDA_VISIBLE_DEVICES=0,1,2"
        echo "   ray start --head --num-gpus=3"
        echo ""
        echo "   # 运行训练"
        echo "   python ray_train_multirun.py \\"
        echo "       --config-dir=. \\"
        echo "       --config-name=image_pusht_diffusion_policy_cnn.yaml \\"
        echo "       --seeds=42,43,44 \\"
        echo "       --monitor_key=test/mean_score"
        echo ""
        ;;
    
    4)
        echo -e "${GREEN}评估命令示例：${NC}"
        echo ""
        echo "1. 下载预训练检查点："
        echo "   mkdir -p data/checkpoints"
        echo "   wget -P data/checkpoints/ \\"
        echo "       https://diffusion-policy.cs.columbia.edu/data/experiments/low_dim/pusht/diffusion_policy_cnn/train_0/checkpoints/epoch=0550-test_mean_score=0.969.ckpt"
        echo ""
        echo "2. 运行评估（需要GPU）："
        echo "   python eval.py \\"
        echo "       --checkpoint data/checkpoints/epoch=0550-test_mean_score=0.969.ckpt \\"
        echo "       --output_dir data/pusht_eval_output \\"
        echo "       --device cuda:0"
        echo ""
        ;;
    
    5)
        echo -e "${GREEN}启动Python交互式环境...${NC}"
        echo "提示: 你可以导入diffusion_policy模块进行测试"
        echo ""
        $HOME/miniconda3/envs/robodiff/bin/python
        ;;
    
    6)
        echo "退出"
        exit 0
        ;;
    
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac
