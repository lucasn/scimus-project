import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import base64
import PIL
from PIL import ImageDraw
from io import BytesIO
from mapping import LABELS_MAPPING
import math

class Visualization:
    def __init__(self, emoji_size=2):
        self.emoji_data = pd.read_csv('./emojiimage-dataset/full_emoji.csv')
        self.emoji_data.set_index('unicode', inplace=True)
        self.emoji_size = emoji_size

        # Placeholder emoji while we don't have the mapping label-emoji
        self.default_emoji = self._retrieve_emoji_as_PIL('U+274C')


    def _retrieve_emoji_as_PIL(self, unicode):
        # We take the data from index 21 to remove unnecessary header
        emoji = self.emoji_data.loc[unicode]['Apple'][21:]
        image_bytes = base64.b64decode(emoji)
        return PIL.Image.open(BytesIO(image_bytes))


    def generate_visualization(self, x, duration, y, labels, figure_name='output.png'):
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

            emoji_unicode = LABELS_MAPPING.get(labels[i])

            if not emoji_unicode:
                ax.imshow(self.default_emoji, 
                        extent=[emoji_x_pos, emoji_x_pos + emoji_width, 
                                emoji_y_pos, emoji_y_pos + emoji_height], 
                        aspect='auto')
            else:
                emoji = self._retrieve_emoji_as_PIL(emoji_unicode.split('-')[1])
                ax.imshow(emoji, 
                        extent=[emoji_x_pos, emoji_x_pos + emoji_width, 
                                emoji_y_pos, emoji_y_pos + emoji_height], 
                        aspect='auto')

        ax.set_xlim(*fig_xlim)  
        ax.set_ylim(*fig_ylim)  

        plt.savefig(figure_name, dpi=300)
    

    def create_emoji_gif(self, labels, output_name='output', frame_size=(200, 200), duration=500):
        frames = []

        for label in labels:
            emoji_image = self.default_emoji

            emoji_unicode = LABELS_MAPPING.get(label)
            if emoji_unicode:
                unicodes = emoji_unicode.split('-')
                for i, unicode in enumerate(unicodes[1:]):
                    if i == 0:
                        emoji_image = self._retrieve_emoji_as_PIL(unicode)
                    else:
                        next_emoji = self._retrieve_emoji_as_PIL(unicode)
                        new_emoji_image = PIL.Image.new("RGBA", (emoji_image.width + next_emoji.width, emoji_image.height)) 
                        new_emoji_image.paste(emoji_image, (0, 0))
                        new_emoji_image.paste(next_emoji, (emoji_image.width, 0))
                        emoji_image = new_emoji_image

            frame = PIL.Image.new("RGBA", frame_size, (255, 255, 255, 1))

            emoji_image = emoji_image.copy().convert("RGBA")
            emoji_image.thumbnail((frame_size[0] // 2, frame_size[1] // 2), PIL.Image.ANTIALIAS)

            x = (frame_size[0] - emoji_image.width) // 2
            y = (frame_size[1] - emoji_image.height) // 2

            frame.paste(emoji_image, (x, y), emoji_image)

            frames.append(frame)

        frames[0].save(
            f'output/{output_name}.gif',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    def create_emoji_circle_gif(self, labels, output_name='output', frame_size=(200, 200), duration=500, circle_radius=70):
        frames = []

        # Convert labels to emoji images
        emoji_images = []
        for label in labels:
            emoji_unicode = LABELS_MAPPING.get(label)
            if emoji_unicode:
                unicodes = emoji_unicode.split('-')
                for i, unicode in enumerate(unicodes[1:]):
                    if i == 0:
                        emoji_image = self._retrieve_emoji_as_PIL(unicode)
                    else:
                        next_emoji = self._retrieve_emoji_as_PIL(unicode)
                        new_emoji_image = PIL.Image.new("RGBA", (emoji_image.width + next_emoji.width, emoji_image.height)) 
                        new_emoji_image.paste(emoji_image, (0, 0))
                        new_emoji_image.paste(next_emoji, (emoji_image.width, 0))
                        emoji_image = new_emoji_image

                emoji_images.append(emoji_image)

        # Determine the number of frames (one for each emoji in the circle)
        num_frames = len(emoji_images)

        frame = PIL.Image.new("RGBA", frame_size, (255, 255, 255, 1))

        # Calculate positions for emojis in a circle
        center_x, center_y = frame_size[0] // 2, frame_size[1] // 2
        angle_step = 2 * math.pi / num_frames

        for i, emoji_image in enumerate(emoji_images):
            frame = frame.copy()
            angle = angle_step * ((i - math.pi / 2))

            emoji_resized = emoji_image.copy().convert("RGBA")
            emoji_resized.thumbnail((frame_size[0] // 6, frame_size[1] // 6), PIL.Image.ANTIALIAS)

            x = center_x + int(circle_radius * math.cos(angle) - emoji_resized.width // 4)
            y = center_y + int(circle_radius * math.sin(angle) - emoji_resized.height // 4)


            # Paste the emoji onto the frame
            frame.paste(emoji_resized, (x - int(0.05*frame_size[0]), y - int(0.05*frame_size[1])), emoji_resized)
            frames.append(frame)

        # Save the frames as a GIF
        frames[0].save(
            f'output/{output_name}_circle.gif',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    def create_emoji_circle_detailled_gif(self, labels, output_name='output', frame_size=(200, 200), duration=500, circle_radius=80):
        frames = []

        # Convert labels to emoji images
        emoji_images = []
        for label in labels:
            _emoji_images = []
            for inside_label in label:
                emoji_unicode = LABELS_MAPPING.get(inside_label)
                if emoji_unicode:
                    unicodes = emoji_unicode.split('-')
                    for i, unicode in enumerate(unicodes[1:]):
                        if i == 0:
                            emoji_image = self._retrieve_emoji_as_PIL(unicode)
                        else:
                            next_emoji = self._retrieve_emoji_as_PIL(unicode)
                            new_emoji_image = PIL.Image.new("RGBA", (emoji_image.width + next_emoji.width, emoji_image.height)) 
                            new_emoji_image.paste(emoji_image, (0, 0))
                            new_emoji_image.paste(next_emoji, (emoji_image.width, 0))
                            emoji_image = new_emoji_image
                _emoji_images.append(emoji_image)
            
            emoji_images.append(_emoji_images)

        # Determine the number of frames (one for each emoji in the circle)
        num_frames = len(emoji_images)

        frame = PIL.Image.new("RGBA", frame_size, (255, 255, 255, 1))

        # Calculate positions for emojis in a circle
        center_x, center_y = frame_size[0] // 2, frame_size[1] // 2
        angle_step = 2 * math.pi / num_frames

        for i in range(num_frames):
            frame = frame.copy()
            angle = angle_step * ((i - math.pi / 2))

            emoji_resized = emoji_images[i][0].copy().convert("RGBA")
            emoji_resized.thumbnail((frame_size[0] // 6, frame_size[1] // 6), PIL.Image.ANTIALIAS)

            x = center_x + int(circle_radius * math.cos(angle) - emoji_resized.width // 4)
            y = center_y + int(circle_radius * math.sin(angle) - emoji_resized.height // 4)

            # Paste the emoji onto the frame
            frame.paste(emoji_resized, (x - int(0.05*frame_size[0]), y - int(0.05*frame_size[1])), emoji_resized)

            emoji_resized2 = emoji_images[i][1].copy().convert("RGBA")
            emoji_resized2.thumbnail((frame_size[0] // 10, frame_size[1] // 10), PIL.Image.ANTIALIAS)

            x = center_x + int(circle_radius/2 * math.cos(angle) - emoji_resized2.width // 4)
            y = center_y + int(circle_radius/2 * math.sin(angle) - emoji_resized2.height // 4)

            # Paste the emoji onto the frame
            frame.paste(emoji_resized2, (x - int(0.05*frame_size[0]), y - int(0.05*frame_size[1])), emoji_resized2)

            frames.append(frame)

        # Save the frames as a GIF
        frames[0].save(
            f'output/{output_name}_circle_detailled.gif',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )

    def create_diagonal_emoji_gif(self, labels, output_name='output', frame_size=(300, 300), duration=1000):
        frames = []

        # Convert labels to emoji images
        emoji_images = []
        for label in labels:
            _emoji_images = []
            for inside_label in label:
                emoji_unicode = LABELS_MAPPING.get(inside_label)
                if emoji_unicode:
                    unicodes = emoji_unicode.split('-')
                    for i, unicode in enumerate(unicodes[1:]):
                        if i == 0:
                            emoji_image = self._retrieve_emoji_as_PIL(unicode)
                        else:
                            next_emoji = self._retrieve_emoji_as_PIL(unicode)
                            new_emoji_image = PIL.Image.new("RGBA", (emoji_image.width + next_emoji.width, emoji_image.height)) 
                            new_emoji_image.paste(emoji_image, (0, 0))
                            new_emoji_image.paste(next_emoji, (emoji_image.width, 0))
                            emoji_image = new_emoji_image
                _emoji_images.append(emoji_image)
            
            emoji_images.append(_emoji_images)

        # Determine the number of frames (one set of 3 emojis per frame)
        num_frames = len(emoji_images)

        for frame_idx in range(num_frames):
            # Create a blank frame
            frame = PIL.Image.new("RGBA", frame_size, (255, 255, 255, 1))

            for emoji_idx, emoji in enumerate(emoji_images[frame_idx]):
                scale_factor = 1 - (emoji_idx * 0.3)
                emoji_resized = emoji.copy().convert("RGBA")
                new_size = (frame_size[0] // 5) * scale_factor
                emoji_resized.thumbnail((new_size, new_size), PIL.Image.ANTIALIAS)

                center_x = (emoji_idx + 1)  * (frame_size[0] // 4)
                center_y =  (emoji_idx + 1) * (frame_size[1] // 4)

                x = center_x - emoji_resized.width
                y = center_y - emoji_resized.height

                frame.paste(emoji_resized, (x, y), emoji_resized)

            frames.append(frame)

        # Save the frames as a GIF
        frames[0].save(
            f'output/{output_name}_diagonal.gif',
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )