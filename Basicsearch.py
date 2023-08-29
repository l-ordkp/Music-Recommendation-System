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

def find_similar_songs_by_name(song_name, num_neighbors=5):
    try:
        song_index = data[data['name'] == song_name].index[0]
    except IndexError:
        print("Song '{}' not found in the dataset.".format(song_name))
        return

    distances, indices = nn_model.kneighbors(normalized_data[song_index].reshape(1, -1), n_neighbors=num_neighbors + 1)

    print("Nearest neighbors for the song '{}':".format(song_name))
    for i in range(1, len(indices[0])):
        neighbor_index = indices[0][i]
        neighbor_song = data.iloc[neighbor_index]
        
        print("{}. Song: {}".format(i, neighbor_song['name']))
        print("   Artist: {}".format(data.iloc[neighbor_index]['artists']))
        print("   Genres: {}".format(data.iloc[neighbor_index]['genres']))
        print("   Year: {}".format(data.iloc[neighbor_index]['year']))
       
# User Interface
while True:
    user_input = input("Enter a song name (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Exiting the recommendation system.")
        break
    
    find_similar_songs_by_name(user_input)

