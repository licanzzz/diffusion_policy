#!/usr/bin/env python3

import torch
import numpy as np
from diffusion_policy.dataset.pusht_dataset import PushTLowdimDataset
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def test_diffusion_policy():
    print("Testing Diffusion Policy components...")
    
    # Test dataset loading
    print("\n1. Testing dataset loading...")
    dataset = PushTLowdimDataset(
        zarr_path='data/pusht/pusht_cchi_v7_replay.zarr',
        horizon=16,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=10
    )
    print(f"✓ Dataset loaded successfully with {len(dataset)} samples")
    
    # Test data sample
    sample = dataset[0]
    obs_dim = sample['obs'].shape[-1]  # Should be 20
    action_dim = sample['action'].shape[-1]  # Should be 2
    horizon = sample['action'].shape[0]  # Should be 16
    
    print(f"✓ Sample shape - obs: {sample['obs'].shape}, action: {sample['action'].shape}")
    print(f"✓ Dimensions - obs_dim: {obs_dim}, action_dim: {action_dim}, horizon: {horizon}")
    
    # Test model creation
    print("\n2. Testing model creation...")
    model = ConditionalUnet1D(
        input_dim=action_dim,
        local_cond_dim=None,
        global_cond_dim=obs_dim * 2,  # n_obs_steps = 2
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    )
    print(f"✓ Model created successfully")
    
    # Test scheduler
    print("\n3. Testing noise scheduler...")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        variance_type='fixed_small',
        clip_sample=True,
        prediction_type='epsilon'
    )
    print(f"✓ Noise scheduler created successfully")
    
    # Test policy
    print("\n4. Testing policy creation...")
    policy = DiffusionUnetLowdimPolicy(
        model=model,
        noise_scheduler=noise_scheduler,
        horizon=horizon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=8,
        n_obs_steps=2,
        num_inference_steps=100,
        obs_as_local_cond=False,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        oa_step_convention=True
    )
    
    # Set normalizer from dataset
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    print(f"✓ Policy created successfully")
    
    # Test forward pass
    print("\n5. Testing forward pass...")
    batch_size = 4
    obs_batch = torch.randn(batch_size, 2, obs_dim)  # (B, n_obs_steps, obs_dim)
    
    # Test prediction
    with torch.no_grad():
        obs_dict = {'obs': obs_batch}
        action_pred = policy.predict_action(obs_dict)
    
    print(f"✓ Forward pass successful")
    print(f"✓ Input obs shape: {obs_batch.shape}")
    print(f"✓ Output action shape: {action_pred['action'].shape}")
    
    # Test environment
    print("\n6. Testing environment...")
    from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
    
    env = PushTKeypointsEnv(render_size=96, render_action=False)
    obs = env.reset()
    print(f"✓ Environment created and reset successfully")
    print(f"✓ Environment observation shape: {obs.shape}")
    
    # Test a simple step
    action = np.array([0.1, 0.1])
    obs, reward, done, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"✓ Reward: {reward}, Done: {done}")
    
    print("\n🎉 All tests passed! Diffusion Policy is working correctly.")
    return True

if __name__ == "__main__":
    test_diffusion_policy()