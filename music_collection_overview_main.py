import os
import music_collection_overview as mco

ANALYSIS_DATA_DIR = 'analysis'
AUDIO_PREDICTIONS_PATH = 'predictions/audio_predictions.json'
DISCOGS_METADATA_PATH = 'metadata/discogs-effnet-bs64-1.json'

def main():

    os.makedirs(ANALYSIS_DATA_DIR, exist_ok=True)

    audio_predictions = mco.load_audio_predictions(AUDIO_PREDICTIONS_PATH)
    genres = mco.load_discogs_music_genres(DISCOGS_METADATA_PATH)
    all_audio_genres = mco.get_audio_genres(audio_predictions, genres)

    print('\n_____MUSIC ANALYSIS REPORT_____\n')

    print('\nGenerating parent genre distribution plot...')
    mco.plot_genre_distribution(genres, all_audio_genres, ANALYSIS_DATA_DIR)

    print('\nGenerating tempo distribution plot...')
    mco.plot_tempo_distribution(audio_predictions, ANALYSIS_DATA_DIR)

    print('\nGenerating danceability distribution plot...')
    mco.plot_danceability_distribution(audio_predictions, ANALYSIS_DATA_DIR)

    threshold = 0.8
    mco.get_genre_danceability_percentage(all_audio_genres, audio_predictions, threshold, 'Rock')
    mco.get_genre_danceability_percentage(all_audio_genres, audio_predictions, threshold, 'Electronic')

    print('\nGenerating key/scale distribution plot...')
    mco.plot_key_scale_distribution(audio_predictions, ANALYSIS_DATA_DIR)
    mco.get_key_scale_data(audio_predictions)

    print('\nGenerating loudness LUFS distribution plot...')
    mco.plot_loudness_distribution(audio_predictions, ANALYSIS_DATA_DIR)

    print('\nGenerating valence vs arousal 2D distribution plot...')
    mco.plot_valence_vs_arousal(audio_predictions, ANALYSIS_DATA_DIR)

    print('\nGenerating voice-instrument classification distribution plot...')
    mco.plot_voice_or_instrument_distribution(audio_predictions, ANALYSIS_DATA_DIR)

    print('\nAll plots and TSV file available in the analysis directory.')

if __name__ == '__main__':
    main()