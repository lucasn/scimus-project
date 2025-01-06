from inference import perform_inference
from utils import read_audio
from visualization import Visualization
from panns_inference import AudioTagging

path_metadata = 'audio/block_length=8+c_method=kmeans+dataset=B+emb=pann+greedy_batch=1+n_clusters=30+n_iter=30+num_block=12+s_method=greedy+s_type=greedy_summary+scen=1+seed_clusters=0+step=summary_summary.csv'
path_audio = 'audio/block_length=8+c_method=kmeans+dataset=B+emb=pann+greedy_batch=1+n_clusters=30+n_iter=30+num_block=12+s_method=greedy+s_type=greedy_summary+scen=1+seed_clusters=0+step=summary_summary.wav'

chunks, times = read_audio(path_audio, path_metadata)

tagger = AudioTagging(checkpoint_path=None, device='cpu')

inferences = perform_inference(tagger, chunks)
visu = Visualization()

higher_scores = [higher_score[1] for higher_score, *_ in inferences]
higher_labels = [higher_score[0] for higher_score, *_ in inferences]
print(higher_labels)
offsets = [offset for offset, _ in times]
duration = [duration for _, duration in times]

visu.generate_visualization(offsets, duration, higher_scores, higher_labels)
