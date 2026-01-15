import numpy as np
import matplotlib.pyplot as plt

def visualize_qtable(q, filename='qtable_visualization.png', title=None):
    lake_map = [
        'SFFFFFFF',
        'FFFFFFFF',
        'FFFHFFFF',
        'FFFFFHFF',
        'FFFHFFFF',
        'FHHFFFHF',
        'FHFFHFHF',
        'FFFHFFFG'
    ]
    
    state_values = np.max(q, axis=1).reshape(8, 8) # max Q-value for each state
    best_actions = np.argmax(q, axis=1).reshape(8, 8) # best action for each state
    
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(state_values, cmap='RdYlGn', interpolation='nearest')
    
    action_arrows = {
        0: '←',  
        1: '↓',  
        2: '→',  
        3: '↑'   
    }
    
    # Draw grid and annotations
    for i in range(8):
        for j in range(8):
            state = i * 8 + j
            cell_type = lake_map[i][j]
            
            # Color the cell background based on type
            if cell_type == 'H':  # Hole
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                          fill=True, color='black', alpha=0.3))
                ax.text(j, i, 'H', ha='center', va='center', 
                       color='red', fontsize=20, fontweight='bold')
            elif cell_type == 'G':  # Goal
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, 
                                          fill=True, color='gold', alpha=0.3))
                ax.text(j, i, 'G', ha='center', va='center', 
                       color='green', fontsize=20, fontweight='bold')
            elif cell_type == 'S':  # Start
                # Show arrow for start
                arrow = action_arrows[best_actions[i, j]]
                ax.text(j, i, arrow, ha='center', va='center', 
                       color='blue', fontsize=30, fontweight='bold')
                ax.text(j, i-0.35, 'S', ha='center', va='center', 
                       color='blue', fontsize=10)
            else:  # Frozen (normal cell)
                # Show best action arrow
                arrow = action_arrows[best_actions[i, j]]
                ax.text(j, i, arrow, ha='center', va='center', 
                       color='black', fontsize=30, fontweight='bold')
            
            # Show state value
            value = state_values[i, j]
            ax.text(j, i+0.35, f'{value:.2f}', ha='center', va='center', 
                   color='darkblue', fontsize=8)
    
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('State Value', rotation=270, labelpad=20)
    
    if title is None:
        title = 'Q-Table Visualization'
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

