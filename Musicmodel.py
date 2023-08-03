import numpy as np
import pandas as pd

# Loading the required libraries, will add the remaining libraries during algorithm implementation
data = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data.csv")
print(data.head())
data_by_artist = pd.read_csv("C:\\Users\\Kshit\\Desktop\\Music Recommendation system\\data_by_artist.csv")
data_w_genres = pd.read_csv("C:\\Users\\Kshit\\Desktop\Music Recommendation system\\data_w_genres.csv")
data_by_year = pd.read_csv("C:\\Users\\Kshit\\Desktop\Music Recommendation system\\data_by_year.csv")
# print(data_by_artist.head())
# print(data_by_year.head())
# print(data_w_genres.head())
print(data.info())
print(data_by_artist.info())
print(data_by_year.info())
print(data_w_genres.info())

 
 
 
 
 