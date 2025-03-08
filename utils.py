import pandas as pd
import numpy as np
import librosa as lb
from mapping import retrieve_blacklist
from inference import interpolate_inference, retrieve_sorted_results
from visualization import Visualization
from sys import exit

def read_audio_csv(path_audio, path_metadata, desired_sample_rate=None):
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
    BLACKLIST = retrieve_blacklist()
    higher_scores = [higher_score[1] for higher_score, *_ in inferences]

    higher_labels = []
    for scores in inferences:
        i = 0
        while scores[i][0] in BLACKLIST:
            i += 1
        higher_labels.append(scores[i][0])

    return higher_scores, higher_labels

def extract_3best_labels(inferences):
    BLACKLIST = retrieve_blacklist()
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

# def read_and_infer_full_audio(path_audio, audio_tagger, desired_sample_rate=None):
#     audio, _ = lb.load(path_audio, sr=desired_sample_rate)
#     clipwise_output, _ = audio_tagger.inference(np.reshape(audio, (1, -1)))
#     inference_result = retrieve_sorted_audio_tagging_results(clipwise_output)
    
#     return inference_result

def read_audio_without_metadata(path_audio, desired_sample_rate=None):
    audio, sr = lb.load(path_audio, sr=desired_sample_rate)

    n_chunks = 12

    duration = lb.get_duration(y=audio, sr=sr)
    chunk_duration = duration / n_chunks

    audio_chunks = []
    audio_times = []
    for i in range(n_chunks): 
        offset = i * chunk_duration
        chunk, sr = lb.load(path_audio, sr=desired_sample_rate, offset=offset, duration=chunk_duration)

        audio_chunks.append((chunk, sr))
        audio_times.append((offset, chunk_duration))

    return audio_chunks, audio_times
        
def concatenate_chunks(chunks, times):
    new_chunks = []
    new_times = []
    sr = chunks[0][1]
    for i in range(0, len(chunks), 2):
        new_chunks.append((lb.util.stack([chunks[i][0], chunks[i+1]][0]), sr))
        new_times.append((times[i][0], times[i][1] + times[i+1][1]))

    return new_chunks, new_times

def read_audio(path_audio):
    window_size = 10
    step_size = 1

    audio, sr = lb.load(path_audio)
    audio_duration = int(lb.get_duration(y=audio, sr=sr))

    offset = 0
    audio_chunks = []
    audio_times = []
    for i in range(audio_duration):
        if offset + window_size < audio_duration:
            chunk, sr = lb.load(path_audio, sr=sr, offset=offset, duration=window_size)
            audio_chunks.append((chunk, sr))
            audio_times.append((offset, window_size))
            offset += step_size
    
    return audio_chunks, audio_times

def generate_circle_from_inference(inferences, number_points, output_name='circle'):
    if number_points != 1:
        interp_inferences = interpolate_inference(inferences, number_points)
    else:
        interp_inferences = np.mean(inferences, axis=0, keepdims=True)

    results = []
    for infer in interp_inferences:
        results.append(retrieve_sorted_results(infer))

    _, best_labels = extract_best_scores(results)

    Visualization().create_emoji_circle_gif(best_labels, output_name=output_name)

    top3_best_labels = extract_3best_labels(results)

    print(f'\n>>> Best results for circle with {number_points} points')
    print(f"{'Chunk':<30}{'Label 1':<30}{'Label 2':<30}{'Label 3':<30}")
    for chunk_idx, labels in enumerate(top3_best_labels):
        print(f'{chunk_idx:<30}{labels[0]:<30}{labels[1]:<30}{labels[2]:<30}')

def print_inference_scores(inferences):
    results = []
    for infer in inferences:
        results.append(retrieve_sorted_results(infer))

    top3_best_labels = extract_3best_labels(results)

    print(f'\n>>> Best results raw inference')
    print(f"{'Chunk':<30}{'Label 1':<30}{'Label 2':<30}{'Label 3':<30}")
    for chunk_idx, labels in enumerate(top3_best_labels):
        print(f'{chunk_idx:<30}{labels[0]:<30}{labels[1]:<30}{labels[2]:<30}')