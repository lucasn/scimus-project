import numpy as np
from panns_inference import labels
from scipy.interpolate import interp1d

def retrieve_sorted_results(clipwise_output):
    """
    Returns a list of tuples containing audio tagging labels and their corresponding 
    output scores, sorted in descending order of the scores.

    Args:
        clipwise_output (ndarray): Array of output scores for each label.

    Returns:
        list of tuple: Each tuple contains a label (str) and its corresponding output score (float), 
        sorted by score in descending order.
    """

    sorted_indexes = np.argsort(clipwise_output)[::-1]
    sorted_indexes = np.reshape(sorted_indexes, -1)

    results = []
    for k in range(len(clipwise_output)):
        results.append((np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))

    return results

def perform_inference(model, audio_chunks):
    """
    Performs inference on an array of audio chunks using an audio tagging model 
    and returns the results sorted by score for each chunk.

    Args:
        model (AudioTagging): An instance of the AudioTagging class from 
            the `panns_inference` library, used to process audio data.
        audio_chunks (list of tuple): List of audio chunks, where each chunk is a tuple 
            containing the audio data (ndarray) and its sample rate (int).

    Returns:
        list of list of tuple: A list of sorted inference results for each chunk. 
        Each inner list contains tuples of labels and scores.
    """
    inference_results = []
    for chunk, _ in audio_chunks:
        clipwise_output, _ = model.inference(np.reshape(chunk, (1, -1)))
        # chunk_inference_result = retrieve_sorted_audio_tagging_results(clipwise_output)
        inference_results.append(clipwise_output.reshape(-1))

    return np.array(inference_results)


def interpolate_inference(inference_original, number_points):
    x_original = np.linspace(0, 1, inference_original.shape[0])
    x_new = np.linspace(0, 1, number_points)

    interp_func = interp1d(x_original, inference_original, axis=0, kind='linear')

    return interp_func(x_new)