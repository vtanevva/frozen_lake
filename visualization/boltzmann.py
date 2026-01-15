import numpy as np

def select_action_boltzmann(q, state, temperature):
    q_values = q[state, :]
    
    # Subtract max to avoid numerical overflow in exp()
    q_values = q_values - np.max(q_values)
    
    # Calculate exponentials
    exp_q = np.exp(q_values / temperature)
    
    # Calculate probabilities (softmax)
    probabilities = exp_q / np.sum(exp_q)
    
    # Sample action according to probabilities
    action = np.random.choice(len(probabilities), p=probabilities)
    
    return action


def select_action_epsilon_greedy(q, state, epsilon, rng):
    """
    Select action using epsilon-greedy exploration policy.
    
    Parameters:
    -----------
    q : numpy array
        Q-table of shape (num_states, num_actions)
    state : int
        Current state
    epsilon : float
        Exploration rate (0 to 1)
        - High epsilon (e.g., 1.0): more exploration
        - Low epsilon (e.g., 0.0): more exploitation
    rng : numpy random generator
        Random number generator
    
    Returns:
    --------
    action : int
        Selected action
    """
    if rng.random() < epsilon:
        # Explore: random action
        action = rng.integers(q.shape[1])
    else:
        # Exploit: best action
        action = np.argmax(q[state, :])
    
    return action

