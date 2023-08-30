import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the pre-trained nearest neighbors model
nn_model = joblib.load("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\musicmodel1.pkl")

# Load the necessary data
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data.csv")
data_w_genres = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data_w_genres.csv")
column_toadd = data_w_genres['genres']
data['genres'] = column_toadd

# Define selected features for similarity calculation
selected_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
                     'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
                     'mode', 'popularity', 'speechiness', 'tempo']

# Encode categorical variables
label_encoder = LabelEncoder()
data['artists_encoded'] = label_encoder.fit_transform(data['artists'].apply(str))
data['name_encoded'] = label_encoder.fit_transform(data['name'].apply(str))

# Combine features and normalize data
combined_features = pd.concat([data[selected_features], data['artists_encoded'], data['name_encoded']], axis=1)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_features)

# Function to find similar songs based on input
def find_similar_songs_by_input(input_value, input_type, num_neighbors=5):
    if input_type == 'artist':
        artist_matches = data[data['artists'].str.contains(input_value, case=False)]
        if artist_matches.empty:
            print("No artists found with a name containing '{}'.".format(input_value))
            return
        artist_name = artist_matches['artists'].iloc[0]  # Use the first match
        song_index = artist_matches.index[0]  # Use the first song of the matching artist
    elif input_type == 'genre':
        genre_matches = data[data['genres'].str.contains(input_value, case=False, na=False)]
        if genre_matches.empty:
            print("No genres found containing '{}'.".format(input_value))
            return
        genre_song_index = genre_matches.index[0]  # Use the first song of the matching genre
        distances, indices = nn_model.kneighbors(normalized_data[genre_song_index].reshape(1, -1), n_neighbors=num_neighbors + 1)
        song_index = indices[0][0]  # Use the nearest neighbor of the genre song
    else:
        print("Invalid input type.")
        return
    
    distances, indices = nn_model.kneighbors(normalized_data[song_index].reshape(1, -1), n_neighbors=num_neighbors + 1)

    print("Similar songs based on your interest in {} '{}':".format(input_type, input_value))
    for i in range(1, len(indices[0])):
        neighbor_song = data.iloc[indices[0][i]]
        print("{}. Song: {}".format(i, neighbor_song['name']))
        print("   Artist: {}".format(neighbor_song['artists']))
        print("   Genres: {}".format(neighbor_song['genres']))
        print("   Year: {}".format(neighbor_song['year']))
        print("   Popularity: {}".format(neighbor_song['popularity']))
        print("   Energy: {}".format(neighbor_song['energy']))
        print("   Instrumentalness: {}".format(neighbor_song['instrumentalness']))
        print("   Liveness: {}".format(neighbor_song['liveness']))
        print("   Tempo: {}".format(neighbor_song['tempo']))
        print("   Acousticness: {}".format(neighbor_song['acousticness']))
        # Add more details as needed

# User Interface
while True:
    print("Choose an option:")
    print("1. Input an artist name")
    print("2. Input a genre name")
    print("3. Exit")
    
    user_choice = input("Enter the number of your choice: ")
    
    if user_choice == '1':
        user_input = input("Enter an artist name: ")
        find_similar_songs_by_input(user_input, 'artist')
    elif user_choice == '2':
        user_input = input("Enter a genre name: ")
        find_similar_songs_by_input(user_input, 'genre')
    elif user_choice == '3':
        print("Exiting the recommendation system.")
        break
    else:
        print("Invalid choice. Please enter a valid option.")
