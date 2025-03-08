from inference import perform_inference
from utils import read_audio
from panns_inference import AudioTagging
from utils import generate_circle_from_inference
import sys

audio_name = sys.argv[1] if len(sys.argv) > 1 else 'A_1'
path_audio = f'new_audios/{audio_name}.mp3'

chunks, times = read_audio(path_audio)
model = AudioTagging(checkpoint_path=None, device='cpu')

inferences = perform_inference(model, chunks)

generate_circle_from_inference(inferences, 1, f'{audio_name}_1')
generate_circle_from_inference(inferences, 6, f'{audio_name}_6')
generate_circle_from_inference(inferences, 12, f'{audio_name}_12')

