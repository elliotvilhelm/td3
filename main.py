#!/usr/bin/env python3
"""
TD3 Main Script - Clean CLI interface for training and testing
"""

import argparse
import sys
from pathlib import Path

from td3 import TD3, train_td3
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image


def record_gif(model_path: str = "models/td3_pendulum", output_path: str = "static/pendulum.gif", duration: int = 5):
    """Record a GIF of the pendulum in action"""
    
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td3 = TD3(state_dim, action_dim, max_action, device)
    
    try:
        td3.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ No trained model found at {model_path}")
        print("Please train the model first with: python main.py train")
        return

    frames = []
    state, _ = env.reset()
    step = 0
    max_steps = duration * 50  # 50 FPS for smooth animation
    
    print(f"Recording {duration} seconds of pendulum action...")
    
    while step < max_steps:
        action = td3.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        frame = env.render()
        frames.append(frame)
        
        state = next_state
        step += 1
        
        if done:
            state, _ = env.reset()
    
    env.close()
    
    if frames:
        print(f"Saving {len(frames)} frames as GIF...")
        frames_pil = [Image.fromarray(frame) for frame in frames]
        frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=20,  # 50ms per frame = 20 FPS
            loop=0
        )
        print(f"✓ GIF saved as {output_path}")
    else:
        print("No frames captured")


def test_model(model_path: str = "models/td3_pendulum", num_episodes: int = 10, render: bool = True, max_steps: int = 500):
    """Test a trained TD3 model"""
    
    env = gym.make('Pendulum-v1', render_mode='human' if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    td3 = TD3(state_dim, action_dim, max_action, device)
    
    try:
        td3.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ No trained model found at {model_path}")
        print("Please train the model first with: python main.py train")
        return

    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while step < max_steps:
            action = td3.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            step += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward:.2f}")
    
    env.close()
    
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\n📊 Results ({num_episodes} episodes):")
    print(f"   Average: {avg_reward:.2f} ± {std_reward:.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, 'b-', alpha=0.7)
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Average: {avg_reward:.2f}')
    plt.title('TD3 Test Results - Pendulum-v1')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('static/td3_test_results.png', dpi=150, bbox_inches='tight')
    print("📈 Plot saved as 'images/td3_test_results.png'")


def main():
    parser = argparse.ArgumentParser(description="TD3 for Pendulum-v1")
    parser.add_argument("mode", choices=["train", "test", "gif"], help="Mode to run")
    
    parser.add_argument("--model", default="models/td3_pendulum", help="Model path (default: models/td3_pendulum)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes (default: 10)")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode (default: 500)")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering during test")
    
    # GIF arguments
    parser.add_argument("--output", default="pendulum.gif", help="Output GIF filename (default: pendulum.gif)")
    parser.add_argument("--duration", type=int, default=5, help="GIF duration in seconds (default: 5)")
    
    # Training arguments
    parser.add_argument("--max-episodes", type=int, default=200, help="Number of training episodes (default: 200)")
    parser.add_argument("--batch-size", type=int, default=100, help="Training batch size (default: 100)")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor γ (default: 0.99)")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate τ (default: 0.005)")
    parser.add_argument("--policy-noise", type=float, default=0.2, help="Policy noise σ (default: 0.2)")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="Noise clipping (default: 0.5)")
    parser.add_argument("--policy-freq", type=int, default=2, help="Policy update frequency (default: 2)")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="Exploration noise (default: 0.1)")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--model-name", default="td3_pendulum", help="Model name for saving (default: td3_pendulum)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🚀 Starting TD3 training...")
        print(f"   Episodes: {args.max_episodes}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Model name: {args.model_name}")
        
        td3, rewards = train_td3(
            max_episodes=args.max_episodes,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            discount=args.discount,
            tau=args.tau,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_freq=args.policy_freq,
            exploration_noise=args.exploration_noise,
            learning_rate=args.learning_rate,
            model_name=args.model_name
        )
        print("✅ Training complete!")
        
    elif args.mode == "test":
        print("🧪 Testing TD3 model...")
        test_model(args.model, args.episodes, not args.no_render, args.max_steps)
        
    elif args.mode == "gif":
        print("📹 Recording pendulum GIF...")
        record_gif(args.model, args.output, args.duration)


if __name__ == "__main__":
    main() 