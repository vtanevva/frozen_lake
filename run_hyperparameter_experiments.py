"""
Script to run hyperparameter experiments for Q-Learning on Frozen Lake.
Varies learning rate and discount factor to analyze their impact.
"""
import gymnasium as gym
import numpy as np
import pickle
import os
from visualization.boltzmann import select_action_epsilon_greedy

def run_experiment(episodes, learning_rate, discount_factor, is_slippery=True, 
                   exploration_method='epsilon', save_dir=None):
    """
    Run a single Q-Learning experiment with specified hyperparameters.
    
    Returns:
        rewards_per_episode: array of rewards for each episode
        q: final Q-table
    """
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=is_slippery, render_mode=None)
    
    q = np.zeros((env.observation_space.n, env.action_space.n))
    
    # Exploration parameters
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        episode_reward = 0
        
        while not terminated and not truncated:
            # Select action
            if exploration_method == 'epsilon':
                action = select_action_epsilon_greedy(q, state, epsilon, rng)
            else:
                action = np.argmax(q[state, :])
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Reward shaping
            if terminated and reward == 0:  # Fell in hole
                reward = -10
            elif terminated and reward == 1:  # Reached goal
                reward = 1
            else:
                reward = -0.01  # Small penalty for each step
            
            # Q-learning update
            q[state, action] = q[state, action] + learning_rate * (
                reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
            )
            
            episode_reward += reward
            state = new_state
        
        # Decay exploration
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        # Store episode reward (1 if reached goal, 0 otherwise)
        if terminated and reward == 1:
            rewards_per_episode[i] = 1
    
    env.close()
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump(rewards_per_episode, f)
        with open(os.path.join(save_dir, 'qtable.pkl'), 'wb') as f:
            pickle.dump(q, f)
    
    return rewards_per_episode, q


def run_learning_rate_experiments():
    """Run experiments varying learning rate."""
    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    discount_factor = 0.9
    episodes = 15000
    
    print("Running Learning Rate Experiments...")
    print("=" * 70)
    
    results = {}
    for lr in learning_rates:
        print(f"Training with learning_rate={lr}, discount_factor={discount_factor}...")
        save_dir = f'comparison/hyperparameters/learning_rate/lr_{lr}'
        rewards, q = run_experiment(episodes, lr, discount_factor, save_dir=save_dir)
        results[lr] = rewards
        print(f"  Final success rate (last 1000 episodes): {np.mean(rewards[-1000:]):.3f}")
    
    return results


def run_discount_factor_experiments():
    """Run experiments varying discount factor."""
    learning_rate = 0.9
    discount_factors = [0.5, 0.7, 0.9, 0.95, 0.99]
    episodes = 15000
    
    print("\nRunning Discount Factor Experiments...")
    print("=" * 70)
    
    results = {}
    for gamma in discount_factors:
        print(f"Training with learning_rate={learning_rate}, discount_factor={gamma}...")
        save_dir = f'comparison/hyperparameters/discount_factor/gamma_{gamma}'
        rewards, q = run_experiment(episodes, learning_rate, gamma, save_dir=save_dir)
        results[gamma] = rewards
        print(f"  Final success rate (last 1000 episodes): {np.mean(rewards[-1000:]):.3f}")
    
    return results


def run_boltzmann_non_slippery():
    """Run Boltzmann exploration experiments in non-slippery environment."""
    from visualization.boltzmann import select_action_boltzmann
    
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, render_mode=None)
    
    temperatures = [0.01, 0.1, 1.0, 5.0, 10.0]
    episodes = 15000
    learning_rate = 0.9
    discount_factor = 0.9
    temp_decay_rate = 0.001
    min_temp = 0.01
    
    print("\nRunning Boltzmann Exploration in Non-Slippery Environment...")
    print("=" * 70)
    
    results = {}
    
    for initial_temp in temperatures:
        print(f"Training with initial_temp={initial_temp}...")
        
        q = np.zeros((env.observation_space.n, env.action_space.n))
        temperature = initial_temp
        rewards_per_episode = np.zeros(episodes)
        
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                action = select_action_boltzmann(q, state, temperature)
                
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                # Reward shaping
                if terminated and reward == 0:
                    reward = -10
                elif terminated and reward == 1:
                    reward = 1
                else:
                    reward = -0.01
                
                # Q-learning update
                q[state, action] = q[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q[new_state, :]) - q[state, action]
                )
                
                state = new_state
            
            # Decay temperature
            temperature = max(temperature - temp_decay_rate, min_temp)
            
            if terminated and reward == 1:
                rewards_per_episode[i] = 1
        
        save_dir = f'comparison/hyperparameters/boltzmann_non_slippery/temp_{initial_temp}'
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'rewards.pkl'), 'wb') as f:
            pickle.dump(rewards_per_episode, f)
        with open(os.path.join(save_dir, 'qtable.pkl'), 'wb') as f:
            pickle.dump(q, f)
        
        results[initial_temp] = rewards_per_episode
        print(f"  Final success rate (last 1000 episodes): {np.mean(rewards_per_episode[-1000:]):.3f}")
    
    env.close()
    return results


if __name__ == '__main__':
    # Run all experiments
    lr_results = run_learning_rate_experiments()
    gamma_results = run_discount_factor_experiments()
    boltzmann_results = run_boltzmann_non_slippery()
    
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)
