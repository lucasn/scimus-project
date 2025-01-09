from inference import perform_inference
from utils import read_audio
from visualization import Visualization
from panns_inference import AudioTagging
from mapping import BLACKLIST

path_metadata = 'audio/audio2.csv'
path_audio = 'audio/audio2.wav'

chunks, times = read_audio(path_audio, path_metadata)

tagger = AudioTagging(checkpoint_path=None, device='cpu')

inferences = perform_inference(tagger, chunks)
visu = Visualization()

higher_scores = [higher_score[1] for higher_score, *_ in inferences]
higher_labels = []
for scores in inferences:
    i = 0
    while scores[i][0] in BLACKLIST:
        i += 1
    higher_labels.append(scores[i][0])

print(higher_labels)
offsets = [offset for offset, _ in times]
duration = [duration for _, duration in times]

visu.create_emoji_gif(higher_labels)
