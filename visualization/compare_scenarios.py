import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from visualization.boltzmann import select_action_boltzmann, select_action_epsilon_greedy


def run_training(episodes, is_slippery, exploration_method='epsilon', 
                 initial_temp=10, temp_decay_rate=0.001):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=is_slippery, render_mode=None)
    
    q = np.zeros((env.observation_space.n, env.action_space.n))
    
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    
    # Exploration parameters
    epsilon = 1
    epsilon_decay_rate = 0.0001
    
    temperature = initial_temp
    min_temp = 0.01
    
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Select action based on exploration method
            if exploration_method == 'epsilon':
                action = select_action_epsilon_greedy(q, state, epsilon, rng)
            elif exploration_method == 'boltzmann':
                action = select_action_boltzmann(q, state, temperature)
            else:
                raise ValueError(f"Unknown exploration method: {exploration_method}")
            
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            # Reward shaping
            if terminated and reward == 0:  # Fell in hole
                reward = -10
            elif terminated and reward == 1:  # Reached goal
                reward = 1
            else:
                reward = -0.01  # Small penalty for each step
            
            # Q-learning update
            q[state, action] = q[state, action] + learning_rate_a * (
                reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
            )
            
            state = new_state
        
        # Decay exploration parameters
        if exploration_method == 'epsilon':
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            if epsilon == 0:
                learning_rate_a = 0.0001
        elif exploration_method == 'boltzmann':
            temperature = max(temperature - temp_decay_rate, min_temp)
        
        if reward == 1:
            rewards_per_episode[i] = 1
    
    env.close()
    
    # Calculate cumulative rewards (rolling window of 100)
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):t+1])
    
    return q, sum_rewards


def plot_comparison(scenarios_data, filename='comparison_plot.png'):
    """
    Plot learning curves for multiple scenarios.
    
    Parameters:
    - scenarios_data: list of tuples (label, sum_rewards, color)
    - filename: output filename
    """
    plt.figure(figsize=(14, 8))
    
    for label, sum_rewards, color in scenarios_data:
        plt.plot(sum_rewards, label=label, linewidth=2, color=color, alpha=0.8)
    
    plt.xlabel('Episodes', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Rewards ', fontsize=12, fontweight='bold')
    plt.title('Slippery vs Non-Slippery & Exploration Methods', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {filename}")


def main():
    """
    Run all three scenarios and create comparison plot.
    """
    episodes = 15000
    
    print("=" * 70)
    print("Starting Q-Learning Comparison")
    print("=" * 70)
    
    # Scenario 1: Non-slippery + Epsilon-greedy
    print("\n[1/3] Training: Non-Slippery + Epsilon-Greedy...")
    q1, rewards1 = run_training(episodes, is_slippery=False, exploration_method='epsilon')
    print(f"      Max cumulative reward: {np.max(rewards1):.2f}")
    print(f"      Final cumulative reward: {rewards1[-1]:.2f}")
    
    # Scenario 2: Slippery + Epsilon-greedy
    print("\n[2/3] Training: Slippery + Epsilon-Greedy...")
    q2, rewards2 = run_training(episodes, is_slippery=True, exploration_method='epsilon')
    print(f"      Max cumulative reward: {np.max(rewards2):.2f}")
    print(f"      Final cumulative reward: {rewards2[-1]:.2f}")
    
    # Scenario 3: Slippery + Boltzmann (temperature 10 -> 0.01)
    print("\n[3/3] Training: Slippery + Boltzmann (τ=1 → 0.01)...")
    q3, rewards3 = run_training(episodes, is_slippery=True, exploration_method='boltzmann',
                                initial_temp=1, temp_decay_rate=0.001)
    print(f"      Max cumulative reward: {np.max(rewards3):.2f}")
    print(f"      Final cumulative reward: {rewards3[-1]:.2f}")
    
    # Create comparison plot
    print("\n" + "=" * 70)
    print("Generating comparison plot...")
    scenarios = [
        ("Non-Slippery + ε-Greedy", rewards1, '#2ecc71'),  # Green
        ("Slippery + ε-Greedy", rewards2, '#3498db'),      # Blue
        ("Slippery + Boltzmann (τ=1→0.01)", rewards3, '#e74c3c')  # Red
    ]
    
    plot_comparison(scenarios, filename='scenario_comparison.png')
    
    print("=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 70)
    print(f"{'Scenario':<40} {'Max Reward':>12} {'Final Reward':>12}")
    print("-" * 70)
    for label, rewards, _ in scenarios:
        print(f"{label:<40} {np.max(rewards):>12.2f} {rewards[-1]:>12.2f}")
    print("-" * 70)


if __name__ == '__main__':
    main()

