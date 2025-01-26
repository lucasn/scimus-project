import pandas as pd
import numpy as np
import librosa as lb
from mapping import BLACKLIST


def read_audio(path_audio, path_metadata, desired_sample_rate=None):
    """
    Reads an audio file and his metadata, extracting specified chunks based on timestamps.

    Args:
        path_audio (str): Path to the audio file.
        path_metadata (str): Path to the metadata CSV file containing 'summary_start' and 'summary_end' columns.
        desired_sample_rate (int, optional): Desired sample rate for the audio. Defaults to the native sample rate.

    Returns:
        tuple: 
            - audio_chunks (list of tuples): List of tuples containing audio chunks and their sample rates.
            - audio_times (list of tuples): List of tuples containing offsets (start times) and durations of the audio chunks.
    """
    metadata = pd.read_csv(path_metadata, sep=',')

    # Converting to datetime format
    metadata['summary_start'] = pd.to_datetime(metadata['summary_start'], infer_datetime_format=True).astype('datetime64[s]')
    metadata['summary_end'] = pd.to_datetime(metadata['summary_end'], infer_datetime_format=True).astype('datetime64[s]')

    audio_chunks = []
    audio_times = []
    for i in range(1, len(metadata) - 1):
        # The subtraction of the datetimes return a deltatime, that we can convert to float
        # by dividing it by timedelta64(1, 's')
        offset = (metadata['summary_start'][i].to_numpy() - np.datetime64('today', 's')) / np.timedelta64(1, 's')    
        duration = ((metadata['summary_end'][i] - metadata['summary_start'][i]).to_numpy() / np.timedelta64(1, 's'))

        chunk, sr = lb.load(path_audio, sr=desired_sample_rate, offset=offset, duration=duration)

        audio_chunks.append((chunk, sr))
        audio_times.append((offset, duration))

    return audio_chunks, audio_times

def extract_best_scores(inferences):
    higher_scores = [higher_score[1] for higher_score, *_ in inferences]

    higher_labels = []
    for scores in inferences:
        i = 0
        while scores[i][0] in BLACKLIST:
            i += 1
        higher_labels.append(scores[i][0])

    return higher_scores, higher_labels

def extract_3best_labels(inferences):
    best_labels = []
    for i in range(len(inferences)):
        _best_labels = []
        count = 0
        j = 0
        while count < 3:
            if inferences[i][j][0] not in BLACKLIST:
                _best_labels.append(inferences[i][j][0])
                count += 1
            j += 1
        best_labels.append(_best_labels)
    
    return best_labels
            
        

