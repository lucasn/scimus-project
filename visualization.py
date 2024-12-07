import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import base64
import PIL
from io import BytesIO

class Visualization:
    def __init__(self, emoji_size=2):
        self.emoji_data = pd.read_csv('./emojiimage-dataset/full_emoji.csv')
        self.emoji_data.set_index('unicode', inplace=True)
        self.emoji_size = emoji_size

        # Placeholder emoji while we don't have the mapping label-emoji
        self.emoji = self._retrieve_emoji_as_PIL('U+1F609')


    def _retrieve_emoji_as_PIL(self, unicode):
        # We take the data from index 21 to remove unnecessary header
        emoji = self.emoji_data.loc[unicode]['Apple'][21:]
        image_bytes = base64.b64decode(emoji)
        return PIL.Image.open(BytesIO(image_bytes))


    def generate_visualization(self, x, duration, y, figure_name='output.png'):
        """
        Generates a bar chart visualization with emoji annotations.

        Args:
            x (list of float): Start points of the bars (sound chunks).
            duration (list of float): Durations of each bar.
            y (list of float): Scores or heights of the bars.
            figure_name (str, optional): Output file name for the saved figure. Defaults to 'output.png'.
        """
        cmap = get_cmap('viridis')  
        colors = [cmap(i / len(x)) for i in range(len(x))]  
        _, ax = plt.subplots()

        fig_xlim = [-1, np.max(np.array(x) + np.array(duration)) + 1]
        fig_ylim = [0, np.max(y) + 0.1]

        x_unit = np.abs(fig_xlim[1] - fig_xlim[0])
        y_unit = np.abs(fig_ylim[1] - fig_ylim[0])

        for i in range(len(x)):
            ax.bar(x[i], y[i], width=duration[i], 
                align='edge', edgecolor='black', 
                color=colors[i], alpha=0.5, zorder=-i)
            
            # Centralizing the emoji in the middle of the bar
            emoji_height = self.emoji_size / x_unit
            emoji_width = self.emoji_size / y_unit
            emoji_x_pos = x[i] + duration[i] / 2 - emoji_width / 2
            # the 1.1 is to adjust the position lower than the top of the bar
            emoji_y_pos = y[i] - 1.1*emoji_height 


            ax.imshow(self.emoji, 
                      extent=[emoji_x_pos, emoji_x_pos + emoji_width, 
                              emoji_y_pos, emoji_y_pos + emoji_height], 
                      aspect='auto')


        ax.set_xlim(*fig_xlim)  
        ax.set_ylim(*fig_ylim)  

        plt.savefig(figure_name, dpi=300)