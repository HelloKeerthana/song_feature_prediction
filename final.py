import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Load dataset
df = pd.read_csv('dataset.csv')

# Selecting a smaller subset
df = df.sample(n=1000, random_state=42)

# Deleting all the empty data point rows
df = df.dropna()

# Encode categorical features
artist_encoder = LabelEncoder()
track_name_encoder = LabelEncoder()
genre_encoder = LabelEncoder()

df['artists'] = artist_encoder.fit_transform(df['artists'])
df['track_name'] = track_name_encoder.fit_transform(df['track_name'])
df['track_genre'] = genre_encoder.fit_transform(df['track_genre'])

# Features and target variables
X = df[['artists', 'track_name']]
y_genre = df['track_genre']

# Train-test split for genre prediction
X_train, X_test, y_train_genre, y_test_genre = train_test_split(X, y_genre, test_size=0.2, random_state=42)

# Train the RandomForest model for genre prediction
rf_model_genre = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_genre.fit(X_train, y_train_genre)

# Function to make predictions based on user input
def predict_features(user_input):
    artist_input = artist_encoder.transform([user_input['artists']])
    track_input = track_name_encoder.transform([user_input['track_name']])

    input_data = np.array([
        artist_input[0], track_input[0]
    ]).reshape(1, -1)

    predicted_genre_encoded = rf_model_genre.predict(input_data)
    predicted_genre = genre_encoder.inverse_transform(predicted_genre_encoded)

    return predicted_genre[0]

# Function to get song details
def get_song_details(artist, track_name):
    song_details = df[(df['artists'] == artist_encoder.transform([artist])[0]) & 
                      (df['track_name'] == track_name_encoder.transform([track_name])[0])]
    if not song_details.empty:
        return song_details.iloc[0]
    return None

# Streamlit UI Setup
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
        font-family: 'Orbitron', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00ffcc;
        text-shadow: 0 0 10px lightgray, 0 0 20px lightgray, 0 0 30px lightgray;
    }
    button {
        background-color: white;
        color: black;
    }
    .stForm {
        background-color: lightgray;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("song feature prediction")
st.write("select the artist and song title, and the model will predict the genre.")

# Initialize session state
if 'artist_selected' not in st.session_state:
    st.session_state.artist_selected = None

# Create options for artist
artist_options = artist_encoder.inverse_transform(np.unique(df['artists']))

# Artist selection
artist = st.selectbox('select artist', artist_options)

# Button to update songs based on selected artist
if st.button("update songs"):
    st.session_state.artist_selected = artist

# Filter track options based on selected artist
if st.session_state.artist_selected:
    artist_encoded = artist_encoder.transform([st.session_state.artist_selected])[0]
    track_options = track_name_encoder.inverse_transform(df[df['artists'] == artist_encoded]['track_name'].unique())
else:
    track_options = []

# Song selection (only if artist is selected)
if st.session_state.artist_selected:
    track_name = st.selectbox('Select Song Title', track_options)
else:
    track_name = None

# Create a form for final prediction
with st.form(key='user_input_form'):
    # Submit button for prediction
    submit_button = st.form_submit_button(label='Predict Features')

# Collect the input features into a dictionary
if submit_button and st.session_state.artist_selected and track_name:
    features = {
        'artists': st.session_state.artist_selected,
        'track_name': track_name
    }

    predicted_genre = predict_features(features)
    song_details = get_song_details(st.session_state.artist_selected, track_name)

    if song_details is not None:
        st.write(f"Predicted Genre: {predicted_genre}")
        st.write(f"Song Name: {track_name}")
        st.write(f"Artist Name: {st.session_state.artist_selected}")
        st.write(f"Popularity: {song_details['popularity']}")
        st.write(f"Tempo: {song_details['tempo']}")
        # Add more details as needed
    else:
        st.write("Song details not found.")