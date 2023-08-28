import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib
nn_model = joblib.load("C:\Users\Kshit\Desktop\Music Recommendation system\musicmodel.pkl")
# Load the necessary data
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data.csv")
data_w_genres = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data_w_genres.csv")

# Combine the selected features with the encoded artist information
selected_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
                     'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
                     'mode', 'popularity', 'speechiness', 'tempo']

label_encoder = LabelEncoder()
data['artists_encoded'] = label_encoder.fit_transform(data['artists'].apply(str))

combined_features = pd.concat([data[selected_features], data['artists_encoded']], axis=1)

# Preprocess and normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_features)

# User input
song_name = input("Enter the name of the song: ")

# Find the index of the entered song in the data
song_index = data_w_genres[data_w_genres['name'] == song_name].index[0]

# Use the KNN model to find the nearest neighbors
distances, indices = nn_model.kneighbors(normalized_data[song_index].reshape(1, -1), n_neighbors=6)

# Print the nearest neighbor songs
print("Nearest neighbors of the song '{}' are:".format(song_name))
for i in range(1, len(indices[0])):
    neighbor_song = data_w_genres.iloc[indices[0][i]]['name']
    print("{}. {}".format(i, neighbor_song))