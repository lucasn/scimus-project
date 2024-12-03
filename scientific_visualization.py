import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import base64
import PIL
from io import BytesIO

x = [0, 0, 2] 
duration = [5, 5, 6] 
y = [2, 4, 6] 


def main():
    visu = Visualization()
    visu.generate_visualization(x, duration, y)


class Visualization:
    def __init__(self, emoji_size=0.8):
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
            
            # Centralizing the emoji in the middle of the bar
            emoji_x_pos = (2*x[i] + duration[i] - self.emoji_size)/2
            emoji_y_pos = y[i] - self.emoji_size*1.1

            ax.imshow(self.emoji, extent=[emoji_x_pos, emoji_x_pos + self.emoji_size, emoji_y_pos, emoji_y_pos + self.emoji_size])


        ax.set_xlim(-1, np.max(np.array(x) + np.array(duration)) + 1)  
        ax.set_ylim(0, np.max(y))  
        ax.set_xticks([0, 2, 5, 8])
        ax.set_yticks(range(0, 9))

        plt.savefig(figure_name)


if __name__ == '__main__':
    main()