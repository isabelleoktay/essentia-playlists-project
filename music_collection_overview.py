import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import os


def load_audio_predictions(filename):
    """
    Load audio predictions from a JSON file.

    Parameters:
        filename (str): The path to the JSON file containing audio predictions.

    Returns:
        dict: A dictionary containing audio predictions.
    """
    with open(filename, "r") as f:
        audio_predictions = json.load(f)

    return audio_predictions


def load_discogs_music_genres(filename):
    """
    Load music genres from a JSON file.

    Parameters:
        filename (str): The path to the JSON file containing music genres.

    Returns:
        list: A list of music genres.
    """
    with open(filename, "r") as f:
        discogs_metadata = json.load(f)

    return discogs_metadata['classes']


def get_audio_genres(audio_predictions, genres):
    """
    Get audio genres from predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions.
        genres (list): A list of music genres.

    Returns:
        dict: A dictionary containing audio genres.
    """
    audio_genres = {}

    for audio_file in audio_predictions:
        music_style_index = np.argmax(audio_predictions[audio_file]['music_styles'])
        genre = genres[music_style_index]
        audio_genres[audio_file] = genre

    return audio_genres


def plot_genre_distribution(genres, audio_genres, download_dir):
    """
    Plot the distribution of music genres and save the plot and genre distribution to files.

    Parameters:
        genres (list): A list of music genres.
        audio_genres (dict): A dictionary containing audio genres.
        download_dir (str): The directory to save the files.
    
    Returns:
        None
    """
    parent_genre_count = {}
    all_genre_count = {}

    # get all genre and parent genre counts
    for genre in genres:
        all_genre_count[genre] = 0
        parent_genre = genre.split('---')[0]

        if parent_genre not in parent_genre_count:
            parent_genre_count[parent_genre] = 0

    for audio_file in audio_genres:
        genre = audio_genres[audio_file]
        parent_genre = genre.split('---')[0]
        
        parent_genre_count[parent_genre] += 1
        all_genre_count[genre] += 1

    print('*****Parent genres present in the collection*****')
    print(json.dumps(parent_genre_count, indent=4))

    genre_df = pd.DataFrame(list(parent_genre_count.items()), columns=['Genre', 'Counts'])

    # get number of unique genre and parents genres
    unique_count_all = sum(1 for _, count in all_genre_count.items() if count > 0)
    unique_count_parent = sum(1 for _, count in parent_genre_count.items() if count > 0)
    print(f'MusAV collection contains {unique_count_all} unique genres.')
    print(f'MusAV collection contains {unique_count_parent} parent genres.')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Genre', y='Counts', data=genre_df, hue='Genre', palette='viridis', legend=False)
    plt.xlabel('Genres', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title('Distribution of Genres in MusAV Dataset', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(download_dir, "parent_genre_distribution.png"))
    plt.show(block=False)

    full_genre_distribution_download_path = os.path.join(download_dir, "full_genre_distribution.tsv")

    # write all_genre_count to a TSV file
    print('Downloading all genre distribution as a TSV file...')
    with open(full_genre_distribution_download_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['Genre', 'Counts'])
        for genre, count in all_genre_count.items():
            writer.writerow([genre, count])


def plot_tempo_distribution(audio_predictions, download_dir):
    """
    Plot the distribution of tempo values (beats per minute) in the audio predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        download_dir (str): The directory where the plot image will be saved.

    Returns:
        None
    """
    tempos = [audio_predictions[audio_file]['tempo'] for audio_file in audio_predictions]
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    sns.histplot(tempos, kde=True, color='darkorchid', fill=True)
    
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Count')
    plt.title('Distribution of Tempos in the MusAV Dataset')
    plt.savefig(os.path.join(download_dir, "tempo_distribution.png"))

    plt.show(block=False)


def plot_danceability_distribution(audio_predictions, download_dir):
    """
    Plot the distribution of danceability scores in the audio predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        download_dir (str): The directory where the plot image will be saved.

    Returns:
        None
    """
    danceability = [audio_predictions[audio_file]['danceability'][0] for audio_file in audio_predictions]

    # calculate percentage of highly danceable tracks
    count_above_09 = sum(1 for score in danceability if score > 0.9)
    total_scores = len(danceability)
    percentage_above_09 = (count_above_09 / total_scores) * 100
    print(f'Total number of highly danceable tracks (danceability > 0.9): {percentage_above_09:.2f}%')
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))  
    sns.histplot(danceability, kde=True, color='skyblue')  

    plt.xlabel('Danceability')
    plt.ylabel('Count')
    plt.title('Distribution of Danceability in the MusAV Dataset')
    plt.savefig(os.path.join(download_dir, "danceability_distribution.png"))
    
    plt.show(block=False)


def plot_key_scale_distribution(audio_predictions, download_dir):
    """
    Plot the distribution of key and scale predictions for each profile in the audio predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        download_dir (str): The directory where the plot images will be saved.

    Returns:
        None
    """
    rows = []
    agreement_count = 0  
    total_count = 0  

    # calculate the number of times all three profiles agree
    for audio_file, values in audio_predictions.items():
        total_count += 1 
        key_profiles = values["key"]
        keys = [key[0] for key in key_profiles.values()]
        scales = [key[1] for key in key_profiles.values()]
        if len(set(keys)) == 1 and len(set(scales)) == 1:
            agreement_count += 1  
        for profile, key_scale in key_profiles.items():
            rows.append({"audio_file": audio_file, "profile": profile, "key": key_scale[0], "scale": key_scale[1]})

    df = pd.DataFrame(rows)

    # calculate the percentage of times all three key profiles agree
    agreement_percentage = (agreement_count / total_count) * 100
    print(f"Percentage of times all three key profiles agree: {agreement_percentage:.2f}%\n")

    sns.set(style="whitegrid")
    custom_palette = sns.color_palette("pastel", n_colors=len(df["key"].unique()) * 2)

    # plotting for each key profile
    for profile, profile_data in df.groupby("profile"):
        plt.figure(figsize=(8, 6))
        profile_data["key_scale"] = profile_data["key"] + " " + profile_data["scale"]
        sns.countplot(data=profile_data, x="key_scale", hue="key_scale", palette=custom_palette)
        print(f'Plotting distribution for {profile}')
        plt.title(f"Distribution of Key and Scale for {profile} in the MusAV Dataset")
        plt.xlabel("Key and Scale")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(download_dir, f'{profile}_distribution.png'))  # Save the plot as an image file
        plt.show(block=False)


def plot_loudness_distribution(audio_predictions, download_dir):
    """
    Plot the distribution of loudness in the audio predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        download_dir (str): The directory where the plot image will be saved.

    Returns:
        None
    """
    loudness = [audio_predictions[audio_file]['loudness'] for audio_file in audio_predictions]
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    sns.histplot(loudness, kde=True, color='lightgreen', fill=True)
    
    plt.xlabel('Loudness (LUFS)')
    plt.ylabel('Count')
    plt.title('Distribution of Loudness LUFS in the MusAV Dataset')
    plt.savefig(os.path.join(download_dir, "loudness_distribution.png"))

    plt.show(block=False)


def plot_valence_vs_arousal(audio_predictions, download_dir):
    """
    Plot a 2D scatterplot of valence vs arousal in the audio predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        download_dir (str): The directory where the plot image will be saved.

    Returns:
        None
    """
    rows = []
    for audio_file, values in audio_predictions.items():
        arousal_valence = values["arousal_and_valence"]
        rows.append({"audio_file": audio_file, "valence": arousal_valence[0], "arousal": arousal_valence[1]})
    df = pd.DataFrame(rows)

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="valence", y="arousal", color="turquoise", s=100)
    plt.title("Valence vs Arousal in the MusAV Dataset")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.tight_layout()
    plt.savefig(os.path.join(download_dir, "valence_vs_arousal_distribution.png"))

    plt.show(block=False)


def plot_voice_or_instrument_distribution(audio_predictions, download_dir):
    """
    Plot the distribution of voice or instrument classification in the audio predictions.

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        download_dir (str): The directory where the plot image will be saved.

    Returns:
        None
    """
    danceability = [audio_predictions[audio_file]['voice_or_instrument'][1] for audio_file in audio_predictions]
    
    sns.set_style("whitegrid")
    
    plt.figure(figsize=(10, 6))  
    sns.histplot(danceability, kde=True, color='pink')  
    
    plt.xlabel('Is Vocal')
    plt.ylabel('Count')
    plt.title('Distribution of Voice-Instrument Classification in the MusAV Dataset')
    plt.savefig(os.path.join(download_dir, "voice_instrument_distribution.png"))
    
    plt.show(block=False)


def get_genre_danceability_percentage(audio_genres, audio_predictions, threshold, genre):
    """
    Calculate the percentage of tracks with high danceability in a specified genre.

    Parameters:
        audio_genres (dict): A dictionary containing audio track identifiers as keys and their corresponding genres as values.
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different predictions.
        threshold (float): The threshold value for danceability, above which a track is considered to have high danceability.
        genre (str): The genre for which the danceability percentage is to be calculated.

    Returns:
        None
    """
    audio_genre_and_danceability = {}
    for audio in audio_genres:
        audio_genre_and_danceability[audio] = {
            'genre': audio_genres[audio].split('---')[0],
            'danceability': audio_predictions[audio]['danceability'][0]
        }

    total_genre_tracks = sum(1 for data in audio_genre_and_danceability.values() if data["genre"] == genre)

    count_above_09 = sum(1 for _, data in audio_genre_and_danceability.items()
                                if data["genre"] == genre and data["danceability"] > threshold)
    
    percentage_electronic_above_09 = (count_above_09 / total_genre_tracks) * 100 if total_genre_tracks > 0 else 0
    print(f'{percentage_electronic_above_09:.2f}% of {genre} tracks have danceability > {threshold}.')


def get_key_scale_data(audio_predictions):
    """
    Calculate the percentage of major and minor scales present in audio predictions based on different profiles. 

    Parameters:
        audio_predictions (dict): A dictionary containing audio predictions data, where keys are audio file identifiers and values are dictionaries with keys for different profiles and their corresponding key predictions.

    Returns:
        None
    """
    profiles = {'temperley', 'krumhansl', 'edma'}

    # iterate over each profile
    for profile in profiles:
        major_count = 0
        minor_count = 0
        
        # iterate over each audio file and count major and minor keys for the profile
        for _, data in audio_predictions.items():
            key = data["key"][profile]
            if key[1] == "major":
                major_count += 1
            elif key[1] == "minor":
                minor_count += 1
        
        # calculate percentages
        total_count = major_count + minor_count
        major_percentage = (major_count / total_count) * 100 if total_count > 0 else 0
        minor_percentage = (minor_count / total_count) * 100 if total_count > 0 else 0

        print(f'{major_percentage:.2f}% major scale, {minor_percentage:.2f}% minor count in profile {profile}.')