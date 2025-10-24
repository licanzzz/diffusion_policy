#!/usr/bin/env python3

"""
演示脚本：解释Diffusion Policy中的media评估视频
"""

import os
os.environ['DISPLAY'] = ':99'

import torch
import numpy as np
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def demonstrate_video_generation():
    print("🎬 Diffusion Policy 评估视频生成演示")
    print("=" * 60)
    
    print("\n📝 什么是media评估视频？")
    print("评估视频是模型在环境中执行任务时的可视化记录，包含：")
    print("1. 🤖 机器人/智能体的动作执行过程")
    print("2. 🎯 任务完成情况（成功/失败）")
    print("3. 📊 环境状态变化")
    print("4. 🔍 模型决策的可视化效果")
    
    print("\n🎥 视频生成过程：")
    print("1. 模型预测动作序列")
    print("2. 环境执行这些动作")
    print("3. 每一步都渲染画面帧")
    print("4. 将帧序列编码为MP4视频")
    
    print("\n🔧 技术实现：")
    print("- 使用VideoRecordingWrapper包装环境")
    print("- H.264编码，可配置FPS和质量")
    print("- 自动保存到outputs/*/media/目录")
    print("- 上传到WandB进行在线查看")
    
    # 创建一个简单的环境来演示
    print("\n🎮 创建PushT环境演示...")
    env = PushTKeypointsEnv(render_size=96, render_action=False)
    
    print("✓ 环境创建成功")
    print(f"✓ 渲染尺寸: 96x96像素")
    print(f"✓ 观测维度: {env.reset().shape}")
    
    # 演示渲染功能
    print("\n🖼️ 演示环境渲染...")
    obs = env.reset()
    frame = env.render(mode='rgb_array')
    
    print(f"✓ 渲染帧尺寸: {frame.shape}")
    print(f"✓ 像素值范围: [{frame.min()}, {frame.max()}]")
    
    # 模拟几步动作
    print("\n🎯 模拟动作执行...")
    total_reward = 0
    for step in range(5):
        # 随机动作（实际训练中是模型预测的动作）
        action = np.random.uniform(-1, 1, size=2) * 50 + 250  # PushT动作范围
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"  步骤 {step+1}: 动作={action[:2].round(1)}, 奖励={reward:.3f}")
        
        if done:
            print(f"  🎉 任务完成！总奖励: {total_reward:.3f}")
            break
    
    print("\n📁 正常训练时的视频文件结构：")
    print("""
    data/outputs/YYYY.MM.DD/HH.MM.SS_method_task/
    ├── media/
    │   ├── abc123def.mp4    # 训练环境视频1
    │   ├── def456ghi.mp4    # 训练环境视频2
    │   ├── ghi789jkl.mp4    # 测试环境视频1
    │   └── jkl012mno.mp4    # 测试环境视频2
    ├── checkpoints/
    └── logs.json.txt
    """)
    
    print("\n🎯 视频内容说明：")
    print("对于PushT任务，视频会显示：")
    print("- 🔵 蓝色圆形：机器人末端执行器位置")
    print("- 🟡 黄色T形块：需要推动的目标物体") 
    print("- 🟢 绿色区域：目标区域")
    print("- ⚪ 白色点：关键点标记")
    print("- 📈 轨迹线：执行器移动路径")
    
    print("\n📊 评估指标：")
    print("- 成功率：T形块是否推入目标区域")
    print("- 奖励值：基于距离和完成度的连续奖励")
    print("- 执行步数：完成任务所需的时间步")
    print("- 轨迹质量：动作的平滑度和效率")
    
    print("\n🔍 视频的用途：")
    print("1. 📈 监控训练进度：观察模型学习效果")
    print("2. 🐛 调试模型行为：发现异常动作模式")
    print("3. 📝 展示结果：向他人演示模型能力")
    print("4. 📊 分析失败案例：理解模型的局限性")
    print("5. 🎯 优化策略：根据视觉反馈改进模型")
    
    print("\n⚙️ 配置选项：")
    print("- n_train_vis: 训练时录制的视频数量")
    print("- n_test_vis: 测试时录制的视频数量")  
    print("- fps: 视频帧率（默认10fps）")
    print("- crf: 视频质量（数值越小质量越高）")
    
    print("\n🎉 总结：")
    print("media文件夹中的MP4视频是模型在环境中执行任务的")
    print("可视化记录，是评估和理解模型性能的重要工具！")

if __name__ == "__main__":
    demonstrate_video_generation()