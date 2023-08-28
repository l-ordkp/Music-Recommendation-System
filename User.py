import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib
nn_model = joblib.load("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\musicmodel1.pkl")
# Load the necessary data
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data.csv")
data_w_genres = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data_w_genres.csv")
column_toadd = data_w_genres['genres']
data['genres'] = column_toadd

# Combine the selected features with the encoded artist information
selected_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
                     'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
                     'mode', 'popularity', 'speechiness', 'tempo']

label_encoder = LabelEncoder()
data['artists_encoded'] = label_encoder.fit_transform(data['artists'].apply(str))
data['name_encoded'] = label_encoder.fit_transform(data['name'].apply(str))

combined_features = pd.concat([data[selected_features],data['artists_encoded'], data['name_encoded']], axis=1)

# Preprocess and normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_features)

def find_similar_songs_by_name(song_name):
    song_index = data[data['name'] == song_name].index[0]
    distances, indices = nn_model.kneighbors(normalized_data[song_index].reshape(1, -1), n_neighbors=6)
    print("Similar songs for the song are '{}':".format(song_name))
    for i in range(1, len(indices[0])):
        neighbor_song = data.iloc[indices[0][i]]['name']
        print("{}. {}".format(i, neighbor_song))

def find_similar_songs_by_genre(genre):
    genre_index = data[data['genres'] == genre].index[0]
    distances, indices = nn_model.kneighbors(normalized_data[genre_index].reshape(1, -1), n_neighbors=6)
    print("Similar songs for the genre '{}':".format(genre))
    for i in range(1, len(indices[0])):
        neighbor_song = data.iloc[indices[0][i]]['name']
        print("{}. {}".format(i, neighbor_song))

def find_similar_songs_by_artist(artist_name):
    artist_index = data[data['artists'] == artist_name].index[0]
    distances, indices = nn_model.kneighbors(normalized_data[artist_index].reshape(1, -1), n_neighbors=6)
    print("Similar songs for the artist '{}':".format(artist_name))
    for i in range(1, len(indices[0])):
        neighbor_song = data.iloc[indices[0][i]]['name']
        print("{}. {}".format(i, neighbor_song))
pass

def main():
    print("Choose an option:")
    print("1. Find similar songs by name")
    print("2. Find similar songs by genre")
    print("3. Find similar songs by artist")
    
    choice = input("Enter your choice (1/2/3): ")

    switch = {
        '1': lambda: find_similar_songs_by_name(input("Enter the song name: ")),
        '2': lambda: find_similar_songs_by_genre(input("Enter the genre: ")),
        '3': lambda: find_similar_songs_by_artist(input("Enter the artist name: "))
    }

    selected_option = switch.get(choice, lambda: print("Invalid choice"))
    selected_option()

if __name__ == "__main__":
    main()