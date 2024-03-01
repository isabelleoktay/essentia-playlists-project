import pandas as pd
import music_collection_overview as mco


ESSENTIA_ANALYSIS_PATH = 'predictions/audio_predictions.json'
DISCOGS_METADATA_PATH = 'metadata/discogs-effnet-bs64-1.json'


def load_genres():
    """
    Loads audio predictions and corresponding music genres.
    
    Returns:
        pandas.DataFrame: DataFrame containing audio genres.
    """
    audio_predictions = mco.load_audio_predictions(ESSENTIA_ANALYSIS_PATH)
    genres = mco.load_discogs_music_genres(DISCOGS_METADATA_PATH)
    audio_genres = {}
    
    for audio in audio_predictions:
        genre_activations = audio_predictions[audio]['music_styles']
        audio_genres[audio] = dict(zip(genres, genre_activations))

    return pd.DataFrame(audio_genres).T


def load_activations():
    """
    Loads audio predictions and extracts key activations.
    
    Returns:
        pandas.DataFrame: DataFrame containing key activations.
    """
    audio_predictions = mco.load_audio_predictions(ESSENTIA_ANALYSIS_PATH)
    df = pd.DataFrame(audio_predictions).T

    # use the edma profile because it has the most even distributions
    edma = df['key'].apply(lambda x: x.get('edma'))
    df['key'] = edma
    df.drop(columns=['music_styles'], inplace=True)

    return df