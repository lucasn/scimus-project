import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib
import numpy as np



x = [0, 0, 2] 
duration = [5, 5, 6] 
y = [2, 4, 6] 
emojis = ['🍎', '🍊', '🍇']


def generate_visualization(x, duration, y):
    """
    The x parameter represents the start point of the sound chunk
    and the y parameter represents the score for each label
    """
    cmap = get_cmap('viridis')  
    colors = [cmap(i / len(x)) for i in range(len(x))]  

    _, ax = plt.subplots()

    for i in range(len(x)):
        ax.bar(x[i], y[i], width=duration[i], 
               align='edge', edgecolor='black', 
               color=colors[i], alpha=0.5, zorder=-i)
        
        ax.text(
            x[i] + 0.1, y[i] - 0.5,  # Posição do emoji (dentro da barra, canto superior esquerdo)
            emojis[i], fontsize=14, zorder=10, fontname='Noto Color Emoji'
        )
        ax.text(0,0, 'adsfasdfasdf', fontsize=14, fontname='Noto Color Emoji')


    ax.set_xlim(-1, np.max(np.array(x) + np.array(duration)) + 1)  
    ax.set_ylim(0, np.max(y))  
    ax.set_xticks([0, 2, 5, 8])
    ax.set_yticks(range(0, 9))

    plt.savefig('test.png')


generate_visualization(x, duration, y)