import os
import uuid
import streamlit as st
import similarities_utils as su


# set up gloabl file paths
m3u_filepaths_dir = 'playlists/similarities_playlists/'
ESSENTIA_EMBEDDINGS_PATH = 'embeddings/audio_embeddings.json'

# make sure playlist directory exists
os.makedirs(m3u_filepaths_dir, exist_ok=True)

st.write('# 💿 Similarity Playlist Generator 💿')
st.write(f'Using analysis data from `{ESSENTIA_EMBEDDINGS_PATH}`.')

# load embeddings and get track names
audio_embeddings = su.load_embeddings()
all_tracks = list(audio_embeddings.index)

st.write('## 🎤 From which track do you want similar tracks?')

track_name = st.selectbox("Select track:", all_tracks)

# compute dot products of embeddings for all tracks
discogs_dot_products = su.compute_dot_products(audio_embeddings, 'discogs_embeddings')
musiCNN_dot_products = su.compute_dot_products(audio_embeddings, 'musiCNN_embeddings')

st.write('## 🔀 Post-process')
playlist_name = st.text_input("✍🏻 Enter playlist name:")

# generate a random playlist name if no playlist name is specified
if not playlist_name:
    playlist_name = str(uuid.uuid4())

if st.button('RUN'):
    st.write('## 🔊 Results')

    # find top 10 most similar tracks
    top_10_discogs_mp3s = list(su.find_top_10_similar_tracks(discogs_dot_products, track_name))
    top_10_musiCNN_mp3s = list(su.find_top_10_similar_tracks(musiCNN_dot_products, track_name))

    discogs_filepath = os.path.join(m3u_filepaths_dir, playlist_name + '_discogs' + '.m3u8')
    musiCNN_filepath = os.path.join(m3u_filepaths_dir, playlist_name + '_musiCNN' + '.m3u8')

    # store the M3U8 playlist.
    su.download_playlists(discogs_filepath, top_10_discogs_mp3s)
    su.download_playlists(musiCNN_filepath, top_10_musiCNN_mp3s)

    # display original tracks and playlists for each set of embeddings
    st.write(f'### 📀 Your track: {track_name}')
    st.audio(track_name, format="audio/mp3", start_time=0)

    col1, col2 = st.columns(2)

    with col1:
        st.write('### 1️⃣ Audio previews for the discogs playlist:')
        st.write(f'Stored discogs similarity playlist (local filepaths) to `{discogs_filepath}`.')
        for mp3 in top_10_discogs_mp3s:
            st.audio(mp3, format="audio/mp3", start_time=0)

    with col2:
        st.write('### 2️⃣ Audio previews for the musiCNN playlist:')
        st.write(f'Stored discogs similarity playlist (local filepaths) to `{musiCNN_filepath}`.')
        for mp3 in top_10_musiCNN_mp3s:
            st.audio(mp3, format="audio/mp3", start_time=0)

