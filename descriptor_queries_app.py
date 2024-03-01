import os
import uuid
import random
import streamlit as st
import pandas as pd
import descriptor_queries_utils as dqu


m3u_filepaths_dir = 'playlists/descriptor_playlists/'
ESSENTIA_ANALYSIS_PATH = 'predictions/audio_predictions.json'

os.makedirs(m3u_filepaths_dir, exist_ok=True)


# sidebar to show which features to search by
st.sidebar.header('🔍 Select features to filter by:')
show_genre = st.sidebar.checkbox("Genre")
show_tempo = st.sidebar.checkbox("Tempo")
show_vocal_instrumental = st.sidebar.checkbox("Vocal/Instrumental")
show_danceability = st.sidebar.checkbox("Danceability")
show_arousal_valence = st.sidebar.checkbox("Arousal/Valence")
show_key_scale = st.sidebar.checkbox("Key/Scale")

# create title area for playlist generate app
st.write('# 📀 Playlist Generator 📀')
st.write('## ⬅️ Use the sidebar to select the features you want to generate your playlists with!')
st.write(f'Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.')
audio_genres = dqu.load_genres()
audio_activations = dqu.load_activations()
audio_analysis_styles = audio_genres.columns
st.write('Loaded audio analysis for', len(audio_genres), 'tracks.')

# initalize genre selection values for later use (when generating playlists)
style_select = 0
style_rank = 0
if show_genre:
    st.write('## 🎧 Genre')

    # get all styles a user wants between a certain range
    style_select = st.multiselect('Select by style activations:', audio_analysis_styles)
    if style_select:
        
        # show the distribution of activation values for the selected styles.
        st.write(audio_genres[style_select].describe())

        style_select_str = ', '.join(style_select)
        style_select_range = st.slider(f'Select tracks with `{style_select_str}` activations within range:', value=[0.5, 1.])

    st.write('### 🔝 Rank')
    style_rank = st.multiselect('Rank by style activations (multiplies activations for selected styles):', audio_analysis_styles, [])

step_size = 0.01

if show_tempo:
    st.write('## 🥁 Tempo')

    # get minimum and maximum tempo from user
    min_tempo = 60.0
    max_tempo = 200.0
    min_tempo_selected, max_tempo_selected = st.slider("Select tempo range (BPM):", min_tempo, max_tempo, (min_tempo, max_tempo), step=step_size)

if show_vocal_instrumental:
    st.write('## 🎤 Vocal/Instrumental')

    # get vocal/instrumental choice from user
    with_vocals = st.checkbox("Music with vocals")

if show_danceability:
    st.write('## 💃🏻 Danceability')

    # get minimum and maximum danceability from user
    min_danceability = 0.0
    max_danceability = 1.0
    min_dance_selected, max_dance_selected = st.slider("Select danceability range:", min_danceability, max_danceability, (min_danceability, max_danceability), step=step_size)

if show_arousal_valence:
    st.write('## 🎭 Arousal and Valence')

    # get minimum and maximum valence and arousal from user
    min_arousal_valence = 1.0
    max_arousal_valence = 9.0
    min_arousal_selected, max_arousal_selected = st.slider("Select arousal range:", min_arousal_valence, max_arousal_valence, (min_arousal_valence, max_arousal_valence), step=step_size)
    min_valence_selected, max_valence_selected = st.slider("Select valence range:", min_arousal_valence, max_arousal_valence, (min_arousal_valence, max_arousal_valence), step=step_size)

if show_key_scale:
    st.write('## 🎼 Key and Scale')

    # get key and scale from user
    keys = sorted(audio_activations['key'].apply(lambda x: x[0]).unique())
    selected_key = st.selectbox("Select key:", keys)
    selected_scale = st.radio("Select scale:", ('major', 'minor'))

st.write('## 🔀 Post-process')
playlist_name = st.text_input("✍🏻 Enter playlist name:")
max_tracks = st.number_input('🧢 Maximum number of tracks (0 for all):', value=0)
shuffle = st.checkbox('🔀 Random shuffle')

if st.button('RUN'):
    st.write('## 🔊 Results')
    mp3s = list(audio_activations.index)

    # first calculate all the selected styles that are within the given activation range
    if style_select:
        audio_analysis_query = audio_genres.loc[mp3s][style_select]

        result = audio_analysis_query
        for style in style_select:
            result = result.loc[result[style] >= style_select_range[0]]

        st.write(result)
        mp3s = list(result.index)

    # rank styles according to given genres by multiplying the weights
    if style_rank:
        audio_analysis_query = audio_genres.loc[mp3s][style_rank]
        audio_analysis_query['RANK'] = audio_analysis_query[style_rank[0]]
        for style in style_rank[1:]:
            audio_analysis_query['RANK'] *= audio_analysis_query[style]
        ranked = audio_analysis_query.sort_values(['RANK'], ascending=[False])
        ranked = ranked[['RANK'] + style_rank]
        mp3s = list(ranked.index)

        st.write('Applied ranking by audio style predictions.')
        st.write(ranked)

    # get tracks with given tempo by finding tracks within tempo range
    if show_tempo:
        prev_result = audio_activations.loc[mp3s]

        current_result = prev_result.loc[(prev_result['tempo'] >= min_tempo_selected) & (prev_result['tempo'] <= max_tempo_selected)]
        mp3s = list(current_result.index)   

    # get tracks that are vocal or instrumental (vocal > 0.5, instrumental is <= 0.5)
    if show_vocal_instrumental:
        prev_result = audio_activations.loc[mp3s]

        voice_scores = prev_result['voice_or_instrument'].apply(lambda x: x[1])
        if with_vocals:
            mp3s = list(voice_scores[voice_scores > 0.5].index)
        else:
            mp3s = list(voice_scores[voice_scores <= 0.5].index)

    # get tracks within given danceability range (using danceable activation)
    if show_danceability:
        prev_result = audio_activations.loc[mp3s]
        danceability_scores = prev_result['danceability'].apply(lambda x: x[0])

        current_result = danceability_scores[(danceability_scores >= min_dance_selected) & (danceability_scores <= max_dance_selected)]
        mp3s = list(current_result.index) 

    # get tracks within given arousal/valence ranges
    if show_arousal_valence:
        prev_result = audio_activations.loc[mp3s]

        valence = prev_result['arousal_and_valence'].apply(lambda x: x[0])
        arousal = prev_result['arousal_and_valence'].apply(lambda x: x[1]) 

        current_result = prev_result[(valence >= min_valence_selected) & (valence <= max_valence_selected) & 
                 (arousal >= min_arousal_selected) & (arousal <= max_arousal_selected)]  
        mp3s = list(current_result.index)

    # get tracks of a given key and scale
    if show_key_scale:
        prev_result = audio_activations.loc[mp3s]
        key_scale = prev_result['key'].apply(lambda x: x[0] == selected_key and x[1] == selected_scale)

        current_result = prev_result[key_scale]
        mp3s = list(current_result.index)

    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')

    if shuffle:
        mp3s = list(mp3s)
        random.shuffle(mp3s)
        st.write('Applied random shuffle.')

    # generate final result dataframe to display final results
    final_result = audio_activations.loc[mp3s]
    if style_select:
        final_result = pd.merge(final_result, audio_genres[style_select], left_index=True, right_index=True)

    # generate random playlist name if user did not enter playlist name
    if not playlist_name:
        playlist_name = str(uuid.uuid4())

    m3u_filepaths_file = os.path.join(m3u_filepaths_dir, playlist_name + '.m3u8')

    # store the M3U8 playlist.
    with open(m3u_filepaths_file, 'w') as f:
        mp3_paths = [os.path.join('..', mp3) for mp3 in mp3s]
        f.write('\n'.join(mp3_paths))

    # conditionally display message depending on whether a playlist was generated or not
    first_results = 10
    if len(mp3s) == 0:
        st.write("🧙‍♀️🚫 Looks like our playlist wizard's magic wand needs a tune-up. How about a different spell?")
    else:
        st.write(final_result)
        st.write(f'Stored M3U playlist (local filepaths) to `{m3u_filepaths_file}`.')

        if len(mp3s) < 10:
            first_results = len(mp3s)

        st.write('🧙‍♀️✅ Voila! Your playlist, crafted by the magical hands of our playlist wizard, is now ready to cast its spell of rhythm and harmony. Let the enchantment begin!')
        st.write(f'Audio previews for the first {first_results} results:')
        for mp3 in mp3s[:10]:
            st.audio(mp3, format="audio/mp3", start_time=0)