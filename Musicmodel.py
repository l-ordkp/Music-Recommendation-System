import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Loading the required libraries, will add the remaining libraries during algorithm implementation
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data.csv")
# print(data.head())
data_by_artist = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data_by_artist.csv")
data_w_genres = pd.read_csv("C:\\Users\\Kshit\\Desktop\Music Recommendation system\\data_w_genres.csv")
data_by_year = pd.read_csv("C:\\Users\\Kshit\\Desktop\Music Recommendation system\\data_by_year.csv")
# print(data_by_artist.head())
# print(data_by_year.head())
# print(data_w_genres.head())
# print(data.info())
# print(data_by_artist.info())
# print(data_by_year.info())
# print(data_w_genres.info())
## Every data is available in the dataset except one column of the dataset column named genre. So we are going to add that column 
column_toadd = data_w_genres['genres']
data['genres'] = column_toadd
 
## Data Visualisation
# For counting the number of genres
genre_counts = data['genres'].value_counts()

# Plot a pie plot
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Genre Distribution')
plt.show()






 
 
 
 