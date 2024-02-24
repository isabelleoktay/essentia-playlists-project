from essentia.standard import *


def get_tempo(audio):
    bpm, _, _, _, _ = RhythmExtractor2013(audio)

    return bpm


def get_key(audio):
    keys_and_scales = {'temperley': None,
                       'krumhans': None,
                       'edma': None}

    for profile in keys_and_scales:
        key, scale = KeyExtractor(audio, profileType=profile)
        keys_and_scales[profile] = [key, scale]

    return keys_and_scales


def get_loudness(audio):
    _, _, integrated_loudness, _ = LoudnessEBUR128(audio)

    return integrated_loudness

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
    stereo_audio, _, num_channels, _, _, _ = AudioLoader(filename)
    mono_audio = MonoMixer(stereo_audio, num_channels)

    return stereo_audio, mono_audio