#!/usr/bin/env python3

"""
Simple demo script for Diffusion Policy
This script demonstrates how to:
1. Load a dataset
2. Create a policy
3. Run inference on the environment
"""

import torch
import numpy as np
import os

# Set environment for headless operation
os.environ['DISPLAY'] = ':99'

from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv

def main():
    print("🚀 Diffusion Policy Simple Demo")
    print("=" * 50)
    
    # 1. Load dataset
    print("\n📊 Loading dataset...")
    dataset = PushTLowdimDataset(
        zarr_path='data/pusht/pusht_cchi_v7_replay.zarr',
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=50  # Use more episodes for better training
    )
    
    obs_dim = 20  # 9*2 keypoints + 2 state
    action_dim = 2
    horizon = 16
    n_obs_steps = 2
    n_action_steps = 8
    
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"✓ Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # 2. Create model and policy
    print("\n🧠 Creating policy...")
    
    # Create the U-Net model
    model = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=obs_dim * n_obs_steps,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    )
    
    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        variance_type='fixed_small',
        clip_sample=True,
        prediction_type='epsilon'
    )
    
    # Create policy
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=n_action_steps,
        n_obs_steps=n_obs_steps,
        num_inference_steps=100,
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        oa_step_convention=True
    )
    
    # Set normalizer
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    
    print(f"✓ Policy created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. Create environment
    print("\n🎮 Creating environment...")
    env = PushTKeypointsEnv(render_size=96, render_action=False)
    
    print("✓ Environment created")
    
    # 4. Run a simple rollout with random policy
    print("\n🎯 Running rollout with untrained policy...")
    
    obs = env.reset()
    print(f"✓ Environment reset, obs shape: {obs.shape}")
    
    # Convert observation to policy format
    # obs is (40,) -> reshape to keypoints (9, 2) and state (2,) -> flatten to (20,)
    keypoints = obs.reshape(2, -1)[0].reshape(-1, 2)[:9]  # First 9 keypoints
    state = np.array([0.5, 0.5])  # Dummy state for demo
    obs_policy = np.concatenate([keypoints.flatten(), state])  # (18 + 2 = 20,)
    
    # Create observation history (n_obs_steps=2)
    obs_history = np.stack([obs_policy, obs_policy])  # (2, 20)
    obs_tensor = torch.from_numpy(obs_history).float().unsqueeze(0)  # (1, 2, 20)
    
    # Predict actions
    with torch.no_grad():
        obs_dict = {'obs': obs_tensor}
        action_pred = policy.predict_action(obs_dict)
        actions = action_pred['action'][0].numpy()  # (n_action_steps, 2)
    
    print(f"✓ Predicted {actions.shape[0]} actions")
    
    # Execute actions in environment
    total_reward = 0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            print(f"✓ Episode finished at step {i+1}")
            break
    
    print(f"✓ Total reward: {total_reward:.3f}")
    
    # 5. Show dataset statistics
    print("\n📈 Dataset statistics:")
    sample_batch = []
    for i in range(min(100, len(dataset))):
        sample_batch.append(dataset[i])
    
    obs_batch = torch.stack([s['obs'] for s in sample_batch])
    action_batch = torch.stack([s['action'] for s in sample_batch])
    
    print(f"✓ Observation range: [{obs_batch.min():.3f}, {obs_batch.max():.3f}]")
    print(f"✓ Action range: [{action_batch.min():.3f}, {action_batch.max():.3f}]")
    print(f"✓ Observation std: {obs_batch.std():.3f}")
    print(f"✓ Action std: {action_batch.std():.3f}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Train the policy using: python train.py --config-name=train_diffusion_unet_lowdim_workspace")
    print("2. Evaluate trained policy using: python eval.py")
    print("3. Collect your own demonstrations using: python demo_pusht.py")

if __name__ == "__main__":
    main()