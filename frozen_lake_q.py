import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from visualization.qtable import visualize_qtable
from visualization.learning_curve import plot_learning_curve
from visualization.boltzmann import select_action_boltzmann, select_action_epsilon_greedy


def run(episodes, render=False, is_training=True, 
        exploration_method='epsilon', initial_temp=10, temp_decay_rate=0.001):

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if (is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Q-table: (64, 4)
    else:
        f = open('comparison/is_slippery/True.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate  
    discount_factor_g = 0.9 # gamma or discount factor

    # Exploration parameters
    epsilon = 1 if is_training else 0
    epsilon_decay_rate = 0.0001 # epsilon decay rate - 1/0.0001 = 10000 steps
    
    temperature = initial_temp if is_training else 0.01  # For Boltzmann
    min_temp = 0.01  # Minimum temperature
    
    rng = np.random.default_rng() # random number generator
    
    
    rewards_per_episode = np.zeros(episodes)
    

    for i in range(episodes):
        state = env.reset()[0] # 0 to 63
        terminated = False  # True if goal is reached or hole is hit
        truncated = False  # True when actions > 200


        while(not terminated and not truncated):
            # Select action based on exploration method
            if exploration_method == 'epsilon':
                if is_training:
                    action = select_action_epsilon_greedy(q, state, epsilon, rng)
                else:
                    action = np.argmax(q[state, :])  # Pure exploitation when not training
            
            elif exploration_method == 'boltzmann':
                action = select_action_boltzmann(q, state, temperature)
            
            else:
                raise ValueError(f"Unknown exploration method: {exploration_method}")
            
            new_state, reward, terminated, truncated, _ = env.step(action)

            if terminated and reward == 0:  # Fell in hole
                reward = -10  # Penalty for hole
            elif terminated and reward == 1:  # Reached goal
                reward = 1  # Keep goal reward
            else:  
                reward = -0.01  # Small penalty for each step (encourages shorter paths)
            
            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action]
                )

            state = new_state

        # Decay exploration parameters
        if is_training:
            if exploration_method == 'epsilon':
                epsilon = max(epsilon - epsilon_decay_rate, 0)
                if epsilon == 0:
                    learning_rate_a = 0.0001
            
            elif exploration_method == 'boltzmann':
                temperature = max(temperature - temp_decay_rate, min_temp)

        if reward == 1:
            rewards_per_episode[i] = 1
    

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):t+1])

    if is_training:
        f = open('comparison/exploration/boltzmann/temp_0.01/frozen_lake_8x8.pkl', 'wb')
        pickle.dump(q, f)   
        f.close()
        
    return q, sum_rewards  



if __name__ == '__main__':
    q, rewards = run(15000, is_training=True, exploration_method='boltzmann', initial_temp=0.01)
    #q, rewards = run(1, is_training=False)
    
    # Plot learning curve
    plot_learning_curve(rewards, 
                       filename='comparison/exploration/boltzmann/temp_0.01/frozen_lake_8x8.png',
                       title='τ=0.01 → 0.01')
    
    # Visualize Q-table
    visualize_qtable(q, 
                    filename='comparison/exploration/boltzmann/temp_0.01/qtable_visualization.png',
                    title='τ=0.01 → 0.01')

















#Q[state, action] = Q[state, action] + α * (reward + γ * max(Q[new_state, :]) - Q[state, action])

    
    
