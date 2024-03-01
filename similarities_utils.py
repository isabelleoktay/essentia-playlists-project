import pandas as pd
import music_collection_overview as mco
import numpy as np
import os

ESSENTIA_EMBEDDINGS_PATH = 'embeddings/audio_embeddings.json'

def load_embeddings():
    """
    Loads audio predictions and corresponding music genres.
    
    Returns:
        pandas.DataFrame: DataFrame containing audio genres.
    """
    audio_embeddings = mco.load_audio_predictions(ESSENTIA_EMBEDDINGS_PATH)

    for audio_file, embeddings in audio_embeddings.items():
        for key, value in embeddings.items():
            audio_embeddings[audio_file][key] = np.mean(value, axis=0)

    df = pd.DataFrame.from_dict(audio_embeddings, orient='index')

    return df


def compute_dot_products(audio_embeddings, embeddings_column):
    """
    Compute dot products between different embeddings within the same column of a DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame containing embeddings.
        column (str): Name of the column containing embeddings.

    Returns:
        dot_products (numpy.ndarray): Dot products between different embeddings within the column.
    """
    embeddings_array = np.array(audio_embeddings[embeddings_column].tolist())

    dot_products = np.dot(embeddings_array, embeddings_array.T)
    np.fill_diagonal(dot_products, np.nan)

    dot_products_df = pd.DataFrame(dot_products, index=audio_embeddings.index, columns=audio_embeddings.index)

    return dot_products_df


def find_top_10_similar_tracks(dot_products_df, track_name):
    """
    Find the top 10 most similar tracks to a given track.

    Parameters:
        dot_products_df (DataFrame): DataFrame of dot products between embeddings.
        track_name (str): Name of the track for which to find similar tracks.

    Returns:
        top_similar_tracks (list): List of top 10 most similar track names.
    """
    track_dot_products = dot_products_df.loc[track_name]
    top_similar_indices = track_dot_products.sort_values(ascending=False).index[0:10]
    top_similar_tracks = top_similar_indices.tolist()

    return top_similar_tracks


def download_playlists(filename, mp3s):
    """
    Write the paths of mp3 files into a specified file.

    Parameters:
        filename (str): The name of the file to write mp3 paths into.
        mp3s (list): A list containing mp3 filenames.

    Returns:
        None
    """
    with open(filename, 'w') as f:
        mp3_paths = [os.path.join('..', mp3) for mp3 in mp3s]
        f.write('\n'.join(mp3_paths))