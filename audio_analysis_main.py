import os
import json
from tqdm import tqdm
import audio_analysis as aa
import essentia


DATA_PATH = 'data'
EMBEDDINGS_DIR = 'embeddings'
PREDICTIONS_DIR = 'predictions'
DISCOGS_EFFNET_METADATA_PATH = 'metadata/discogs-effnet-bs64-1.json'


def main():
    
    essentia.log.warningActive = False

    # ensure that necessary directories exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    # get all audio file paths
    audio_files = aa.compile_audio_files(DATA_PATH)
    print(f'Found {len(audio_files)} audio files to analyze. Analyzing now...')

    # initialize directories that we will store the audio embeddings and predictions in
    audio_embeddings = {}
    audio_predictions = {}
    for filename in tqdm(audio_files, desc='Loading audio files'):

        # load audio in all necessary versions
        stereo_audio, mono_audio, resampled_mono_audio, _ = aa.load_audio(filename=filename)

        # get signal processing features
        tempo = aa.get_tempo(mono_audio)
        key = aa.get_key(mono_audio)
        loudness = aa.get_loudness(stereo_audio)

        # get features based on machine learning models
        discogs_embeddings, musiCNN_embeddings = aa.get_embeddings(resampled_mono_audio)
        music_styles = aa.get_music_styles(discogs_embeddings)
        voice_or_instrument = aa.classify_voice_or_instrument(discogs_embeddings)
        danceability = aa.get_danceability(discogs_embeddings)
        arousal_and_valence = aa.get_arousal_and_valence(musiCNN_embeddings)

        # store embeddings in one dictionary and features (predictions) in another
        audio_embeddings[filename] = {
            'discogs_embeddings': discogs_embeddings,
            'musiCNN_embeddings': musiCNN_embeddings,
        }

        audio_predictions[filename] = {
            'tempo': tempo,
            'key': key,
            'loudness': loudness,
            'music_styles': music_styles,
            'voice_or_instrument': voice_or_instrument,
            'danceability': danceability,
            'arousal_and_valence': arousal_and_valence
        }

    # donwload features and embeddings
    audio_embeddings_json_path = os.path.join(EMBEDDINGS_DIR, "audio_embeddings.json")
    audio_predictions_json_path = os.path.join(PREDICTIONS_DIR, "audio_predictions.json")

    audio_embeddings_json = aa.convert_numpy_to_list(audio_embeddings)
    audio_predictions_json = aa.convert_numpy_to_list(audio_predictions)

    with open(audio_embeddings_json_path, "w") as json_file:
        json.dump(audio_embeddings_json, json_file)

    with open(audio_predictions_json_path, "w") as json_file:
        json.dump(audio_predictions_json, json_file)


if __name__ == '__main__':
    main()
