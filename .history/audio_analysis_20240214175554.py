import essentia.standard as es


def get_tempo(audio):
    bpm, _, _, _, _ = es.RhythmExtractor2013(audio)

    return bpm


def get_key(audio):
    keys_and_scales = {'temperley': None,
                       'krumhans': None,
                       'edma': None}

    for profile in keys_and_scales:
        key, scale = es.KeyExtractor(audio, profileType=profile)
        keys_and_scales[profile] = [key, scale]

    return keys_and_scales


def get_loudness(audio):
    _, _, integrated_loudness, _ = es.LoudnessEBUR128(audio)

    return integrated_loudness

def get_embeddings(audio):
    discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    discogs_embeddings = discogs_model(audio)

    musiCNN_model = es.TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")
    musiCNN_embeddings = musiCNN_model(audio)

    return discogs_embeddings, musiCNN_embeddings


def get_music_styles(embeddings):
    music_styles_model = es.TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    music_style_predictions = music_styles_model(embeddings)

    return music_style_predictions


def classify_voice_or_instrument(embeddings):
    voice_or_instrument_model = es.TensorflowPredict2D(graphFilename="voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
    voice_or_instrument_predictions = voice_or_instrument_model(embeddings)

    return voice_or_instrument_predictions
 
def get_danceability(audio):
    pass

def get_arousal_and_valence(audio):
    pass

def load_audio_mono(filename):
    stereo_audio, _, num_channels, _, _, _ = es.AudioLoader(filename)
    mono_audio = es.MonoMixer(stereo_audio, num_channels)

    return stereo_audio, mono_audio