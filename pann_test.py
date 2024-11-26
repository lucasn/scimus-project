import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_sound_event_detection_result(framewise_output):
    """Visualization of sound event detection result. 

    Args:
      framewise_output: (time_steps, classes_num)
    """
    out_fig_path = './sed_result.png'

    classwise_output = np.max(framewise_output, axis=0) # (classes_num,)

    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    lines = []
    for idx in idxes:
        line, = plt.plot(framewise_output[:, idx], label=ix_to_lb[idx])
        lines.append(line)

    plt.legend(handles=lines)
    plt.xlabel('Frames')
    plt.ylabel('Probability')
    plt.ylim(0, 1.)
    plt.savefig(out_fig_path)
    print('Save fig to {}'.format(out_fig_path))


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """

    clipwise_output = np.reshape(clipwise_output, -1)

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    sorted_indexes = np.reshape(sorted_indexes, -1)

    # Print audio tagging top probabilities
    for k in range(10):
        print(f'{np.array(labels)[sorted_indexes[k]]}: {clipwise_output[sorted_indexes[k]]:.3f}')


# audio_path = 'audioset_tagging_cnn/resources/R9_ZSCveAHg_7s.wav'
audio_path = 'audio/block_length=8+c_method=kmeans+dataset=A+emb=pann+greedy_batch=1+n_clusters=30+n_iter=30+num_block=12+s_method=greedy+s_type=greedy_summary+scen=1+seed_clusters=0+step=summary_summary.wav'
audio, sr = librosa.core.load(audio_path, sr=None, mono=True)
# print(audio.shape)
audio = audio[None, :]  # (batch_size, segment_samples)
# print(audio.shape)



print('------ Audio tagging ------')
at = AudioTagging(checkpoint_path=None, device='cuda')
(clipwise_output, embedding) = at.inference(audio)

print_audio_tagging_result(clipwise_output)

print('------ Sound event detection ------')
sed = SoundEventDetection(
    checkpoint_path=None, 
    interpolate_mode='nearest', # 'nearest'
)
framewise_output = sed.inference(audio)
"""(batch_size, time_steps, classes_num)"""

plot_sound_event_detection_result(framewise_output[0])