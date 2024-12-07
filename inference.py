import numpy as np
from panns_inference import labels

def retrieve_sorted_audio_tagging_results(clipwise_output):
    """
    Returns a list of tuples containing audio tagging labels and their corresponding 
    output scores, sorted in descending order of the scores.

    Args:
        clipwise_output (ndarray): Array of output scores for each label.

    Returns:
        list of tuple: Each tuple contains a label (str) and its corresponding output score (float), 
        sorted by score in descending order.
    """

    clipwise_output = np.reshape(clipwise_output, -1)
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    sorted_indexes = np.reshape(sorted_indexes, -1)

    results = []
    for k in range(len(clipwise_output)):
        results.append((np.array(labels)[sorted_indexes[k]], clipwise_output[sorted_indexes[k]]))

    return results

def perform_inference(audio_tagger, audio_chunks):
    """
    Performs inference on an array of audio chunks using an audio tagging model 
    and returns the results sorted by score for each chunk.

    Args:
        audio_tagger (AudioTagging): An instance of the AudioTagging class from 
            the `panns_inference` library, used to process audio data.
        audio_chunks (list of tuple): List of audio chunks, where each chunk is a tuple 
            containing the audio data (ndarray) and its sample rate (int).

    Returns:
        list of list of tuple: A list of sorted inference results for each chunk. 
        Each inner list contains tuples of labels and scores.
    """
    inference_results = []
    for chunk, _ in audio_chunks:
        clipwise_output, _ = audio_tagger.inference(np.reshape(chunk, (1, -1)))
        chunk_inference_result = retrieve_sorted_audio_tagging_results(clipwise_output)
        inference_results.append(chunk_inference_result)

    return inference_results