import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(sum_rewards, filename, title=None):
    plt.figure(figsize=(12, 8))
    plt.plot(sum_rewards, linewidth=2)
    
    if title is None:
        title = 'Learning Curve'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

