"""
Script to run epsilon-greedy experiment in slippery environment.
Generates rewards data for epsilon-greedy in slippery Frozen Lake 8x8.
"""
import gymnasium as gym
import numpy as np
import pickle
import os
from visualization.boltzmann import select_action_epsilon_greedy


def run_epsilon_greedy_slippery():
    """Run epsilon-greedy experiment in slippery environment."""
    
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode=None)
    
    episodes = 15000
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    
    rng = np.random.default_rng()
    
    print("\nRunning Epsilon-Greedy Exploration in Slippery Environment...")
    print("=" * 70)
    print(f"Training with epsilon decay, learning_rate={learning_rate}, discount_factor={discount_factor}...")
    
    q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            action = select_action_epsilon_greedy(q, state, epsilon, rng)
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if terminated and reward == 0:
                reward = -10
            elif terminated and reward == 1:
                reward = 1
            else:
                reward = -0.01
            
            q[state, action] = q[state, action] + learning_rate * (
                reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
            )
            
            state = new_state
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        if terminated and reward == 1:
            rewards_per_episode[i] = 1
        
        if (i + 1) % 1000 == 0:
            success_rate = np.mean(rewards_per_episode[max(0, i-999):i+1])
            print(f"  Episode {i+1}/{episodes}: Success rate (last 1000) = {success_rate:.3f}")
    
    env.close()
    
    save_dir = 'comparison/exploration/epsilon-greedy'
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'rewards.pkl'), 'wb') as f:
        pickle.dump(rewards_per_episode, f)
    with open(os.path.join(save_dir, 'qtable.pkl'), 'wb') as f:
        pickle.dump(q, f)
    
    final_success_rate = np.mean(rewards_per_episode[-1000:])
    print(f"\n  Final success rate (last 1000 episodes): {final_success_rate:.3f}")
    print(f"  Results saved to: {save_dir}")
    print("=" * 70)
    
    return rewards_per_episode, q


if __name__ == '__main__':
    rewards, q = run_epsilon_greedy_slippery()
    print("\nEpsilon-greedy slippery experiment completed!")