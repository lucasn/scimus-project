from inference import perform_inference
from utils import read_audio
from visualization import Visualization
from panns_inference import AudioTagging
from utils import extract_best_scores, extract_3best_labels

audio_name = 'B_1'
path_metadata = f'new_audios/{audio_name}.csv'
path_audio = f'new_audios/{audio_name}.mp3'

chunks, times = read_audio(path_audio, path_metadata)

tagger = AudioTagging(checkpoint_path=None, device='cpu')

inferences = perform_inference(tagger, chunks)
visu = Visualization()

duration = [duration for _, duration in times]
offsets = [offset for offset, _ in times]

scores, best_labels = extract_best_scores(inferences)
best_labels3 = extract_3best_labels(inferences)

print('Offset    Duration    Score    Label1    Label2    Label3')
for i in range(len(best_labels)):
    print(f'{offsets[i]}    {duration[i]}    {scores[i]}    {best_labels3[i][0]}    {best_labels3[i][1]}    {best_labels3[i][2]}')

visu.create_emoji_gif(best_labels, output_name=audio_name)
visu.create_emoji_circle_gif(best_labels, output_name=audio_name)
visu.create_diagonal_emoji_gif(best_labels3, output_name=audio_name)
