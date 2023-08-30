**Music Recommendation System**

The Music Recommendation System is a Python project that offers song recommendations based on either artist names or genre names. The system employs the Nearest Neighbors algorithm to identify similar songs and presents relevant details to the user. The project provides an interactive command-line interface for ease of use.

**Features**

Recommends songs similar to the user's input of artist names or genre names.
Displays comprehensive information about recommended songs, including artist, genre, year, and more.
Offers an interactive command-line interface for user engagement.
Utilizes a pre-trained Nearest Neighbors model for efficient similarity calculations.
Installation

Clone the repository using git clone.
Install required dependencies via pip install -r requirements.txt.
Download the pre-trained Nearest Neighbors model from a provided link and place it in the project directory.
Usage

Run the recommendation system using python recommendation_system.py in the command line. The system will prompt you to select between artist and genre inputs. Upon your input, it will present a list of similar songs along with their relevant details.

**Code Structure**
The code is organized into three main parts:

Loading the Model and Data: Imports necessary libraries, loads the pre-trained Nearest Neighbors model, and loads data from CSV files.

Data Preprocessing and Feature Engineering: Encodes categorical variables (artist and song names) and normalizes features for similarity calculations.

Recommendation Algorithm and User Interface: Implements the core recommendation functionality. It defines functions to find similar songs based on user input (artist or genre) and provides an interactive command-line interface for user interaction.

**Contributing**

Contributions are encouraged! Feel free to contribute by opening issues or submitting pull requests if you encounter issues or have ideas for improvements.
