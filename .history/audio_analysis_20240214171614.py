import essentia as es

def get_tempo(audio):
    pass

def get_key(audio):
    pass

def get_loudness(audio):
    pass

def get_embeddings(audio):
    pass

def get_music_styles(audio):
    pass

def classify_voice_or_instrument(audio):
    pass

def get_danceability(audio):
    pass

def get_arousal_and_valence(audio):
    pass

def load_audio_mono(filename):
    stereo_audio, _, num_channels, _, _, _ = es.AudioLoader(filename)
    mono_audio = es.MonoMixer(multi_channel_audio, num_channels)

    return stereo_audio, mono_audio

def parse_mono_to_stereo(audio):
    pass