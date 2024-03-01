import essentia.standard as es
import numpy as np
import os


def get_tempo(mono_audio):
    """
    Estimate the tempo of the input mono audio.

    Parameters:
        mono_audio (numpy.ndarray): Mono audio data.

    Returns:
        float: Estimated tempo in beats per minute (BPM).
    """
    try: 
        bpm, _, _, _, _ = es.RhythmExtractor2013()(mono_audio)
        return bpm
    
    except Exception as e:
        print(f"Error in get_tempo: {e}")
        return None


def get_key(mono_audio):
    """
    Determine the key and scale of the input mono audio using different key extraction profiles.

    Parameters:
        mono_audio (numpy.ndarray): Mono audio data.

    Returns:
        dict: A dictionary containing key and scale information for different key extraction profiles.
    """
    keys_and_scales = {'temperley': None,
                       'krumhansl': None,
                       'edma': None}
    try:
        for profile in keys_and_scales:
            key, scale, _ = es.KeyExtractor(profileType=profile)(mono_audio)
            keys_and_scales[profile] = [key, scale]

        return keys_and_scales
    
    except Exception as e:
        print(f"Error in get_key: {e}")
        return None


def get_loudness(stereo_audio):
    """
    Calculate the integrated loudness of stereo audio.

    Parameters:
        stereo_audio (numpy.ndarray): Stereo audio data.

    Returns:
        float: Integrated loudness value.
    """
    try:
        _, _, integrated_loudness, _ = es.LoudnessEBUR128()(stereo_audio)

        return integrated_loudness
    
    except Exception as e:
        print(f"Error in get_loudness: {e}")
        return None


def get_embeddings(mono_audio):
    """
    Generate embeddings for audio samples using two different models.

    Parameters:
        mono_audio (numpy.ndarray): Mono audio data.

    Returns:
        tuple: A tuple containing Discogs embeddings and MusiCNN embeddings.
    """
    discogs_embeddings = None
    musiCNN_embeddings = None

    try:
        discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        discogs_embeddings = discogs_model(mono_audio)
    except Exception as e:
        print(f"Error in get_discogs_embeddings: {e}")

    try:
        musiCNN_model = es.TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
        musiCNN_embeddings = musiCNN_model(mono_audio)
    except Exception as e:
        print(f"Error in get_musiCNN_embeddings: {e}")

    return discogs_embeddings, musiCNN_embeddings


def get_music_styles(discogs_embeddings):
    """
    Predict the music styles based on Discogs embeddings.

    Parameters:
        discogs_embeddings (numpy.ndarray): Discogs embeddings for audio samples.

    Returns:
        numpy.ndarray: Mean predictions for music styles.
    """
    try:
        discogs_model = es.TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
        discogs_predictions = discogs_model(discogs_embeddings)
        discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

        return discogs_mean_predictions
    
    except Exception as e:
        print(f"Error in get_music_styles: {e}")
        return None


def classify_voice_or_instrument(discogs_embeddings):
    """
    Classify whether the audio contains voice or instrumental music based on Discogs embeddings.

    Parameters:
        discogs_embeddings (numpy.ndarray): Discogs embeddings for audio samples.

    Returns:
        numpy.ndarray: Mean predictions for voice or instrumental classification.
    """
    try:
        discogs_model = es.TensorflowPredict2D(graphFilename="models/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
        discogs_predictions = discogs_model(discogs_embeddings)
        discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

        return discogs_mean_predictions
    
    except Exception as e:
        print(f"Error in classify_voice_or_instrument: {e}")
        return None

 
def get_danceability(discogs_embeddings):
    """
    Calculate the mean danceability predictions from Discogs embeddings.

    Parameters:
        discogs_embeddings (numpy.ndarray): Discogs embeddings for audio samples.

    Returns:
        numpy.ndarray: Mean danceability predictions.
    """
    try:
        discogs_model = es.TensorflowPredict2D(graphFilename="models/danceability-discogs-effnet-1.pb", output="model/Softmax")
        discogs_predictions = np.array(discogs_model(discogs_embeddings))
        discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

        return discogs_mean_predictions
    
    except Exception as e:
        print(f"Error in get_danceability: {e}")
        return None


def get_arousal_and_valence(musiCNN_embeddings):
    """
    Calculate the mean arousal and valence predictions from MusiCNN embeddings.

    Parameters:
        musiCNN_embeddings (numpy.ndarray): MusiCNN embeddings for audio samples.

    Returns:
        numpy.ndarray: Mean arousal and valence predictions.
    """
    try:
        musiCNN_model = es.TensorflowPredict2D(graphFilename="models/emomusic-msd-musicnn-2.pb", output="model/Identity")
        musiCNN_predictions = musiCNN_model(musiCNN_embeddings)
        musiCNN_mean_predictions = np.mean(musiCNN_predictions, axis=0)

        return musiCNN_mean_predictions
    
    except Exception as e:
        print(f"Error in get_arousal_and_valence: {e}")
        return None


def load_audio(filename):
    """
    Load stereo audio and converts it to mono if stereo audio is available.

    Parameters:
        filename (str): The path to the audio file.

    Returns:
        tuple: A tuple containing the stereo audio and the mono audio.
    """
    stereo_audio = None
    mono_audio = None
    resampled_mono_audio = None
    sr = None

    try:
        stereo_audio, sr, num_channels, _, _, _ = es.AudioLoader(filename=filename)()
    except Exception as e:
        print(f"Error in loading stereo audio: {e}")

    try:
        if stereo_audio is not None:
            mono_audio = es.MonoMixer()(stereo_audio, num_channels)
    except Exception as e:
        print(f"Error in converting stereo to mono: {e}")

    try: 
        resampled_mono_audio = es.Resample(inputSampleRate=float(sr), outputSampleRate=float(16000), quality=1).compute(mono_audio)
    except Exception as e:
        print(f"Error in resampling mono audio: {e}")

    return stereo_audio, mono_audio, resampled_mono_audio, sr


def compile_audio_files(data_home):
    """
    Search through a specified directory and its subdirectories to compile a list of audio files. 
    Supported audio file formats include .wav, .mp3, .ogg, .flac, and .aac.

    Parameters:
        data_home (str): The path to the directory where audio files will be searched.

    Returns:
        list: A list containing the absolute paths of all audio files found within the specified directory and its subdirectories.
    """
    audio_files = []
    for root, _, files in os.walk(data_home):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac', '.aac')):  
                audio_file = os.path.join(root, file)
                audio_files.append(audio_file)
    return audio_files


def convert_numpy_to_list(obj):
    """
    Recursively convert NumPy arrays to lists within a nested dictionary.

    Parameters:
        obj (object): The input object to be converted. Can be a NumPy array, a dictionary, or any other object.

    Returns:
        object: The converted object with NumPy arrays replaced by lists.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj