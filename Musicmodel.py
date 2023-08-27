import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import joblib


# Loading the required libraries, will add the remaining libraries during algorithm implementation
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data.csv")
# print(data.head())
data_by_artist = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data_by_artist.csv")
data_w_genres = pd.read_csv("C:\\Users\\Kshit\\Desktop\Music Recommendation system\\data_w_genres.csv")
data_by_year = pd.read_csv("C:\\Users\\Kshit\\Desktop\Music Recommendation system\\data_by_year.csv")

# print(data.info())
# print(data_by_artist.info())
# print(data_by_year.info())
# print(data_w_genres.info())
## Every data is available in the dataset named data except one column of the dataset data_w_genres column named genre. So we are going to add that column 
column_toadd = data_w_genres['genres']
data['genres'] = column_toadd
 
# # ## Data Visualisation
# # For counting the number of genres
genre_counts = data['genres'].value_counts()

# # # Plot a pie plot
# plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.title('Genre Distribution')
# plt.show()

# ## Explicit count
# explicit_counts = data['explicit'].value_counts()
# print(explicit_counts)

## Implimenting K- Means
selected_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 
                     'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 
                     'mode', 'popularity', 'speechiness', 'tempo']

label_encoder = LabelEncoder()
data['artists_encoded'] = label_encoder.fit_transform(data['artists'].apply(str))

# Combine the selected features with the encoded artist information
combined_features = pd.concat([data[selected_features], data['artists_encoded']], axis=1)

# Preprocess and normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(combined_features)

# Fit Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
nn_model.fit(normalized_data)

# Save the model to a file using joblib
joblib.dump(nn_model, 'musicmodel.pkl')



 
 
 
 