<<<<<<< HEAD
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from scipy.io import wavfile

# Load the trained model
model_path = 'C:/Users/kingk/Downloads/pbd2/saved_models/weights.best.basic_mlp.hdf5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess audio and extract MFCC features
def preprocess_audio(audio_path):
    try:
        # Load audio using librosa
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast') 
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Calculate mean of MFCCs
        
        # You can perform additional preprocessing steps or feature scaling here
        
        return mfccs_mean  # Return preprocessed audio features
    
    except Exception as e:
        print("Error encountered while processing audio:", e)
        return None

# Function to make audio predictions
def predict_sound(audio):
    # Preprocess the audio
    processed_audio = preprocess_audio(audio)

    if processed_audio is not None:
        # Make prediction
        prediction = model.predict(np.expand_dims(processed_audio, axis=0))
        return prediction
    else:
        return None

# Streamlit app
st.title('Predict Sound from Audio')
st.write('Upload an audio file to predict its type!')

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Display the uploaded file info
    st.write('Uploaded Audio File:', uploaded_file.name)

    # Make a prediction
    prediction = predict_sound(uploaded_file)
    if prediction is not None:
        class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
        ]
        predicted_class = class_names[np.argmax(prediction)]

        st.write(f"Predicted Sound: {predicted_class}")
    else:
        st.write("Error processing audio. Please upload a valid WAV file.")

=======
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from scipy.io import wavfile

# Load the trained model
model_path = 'C:/Users/kingk/Downloads/pbd2/saved_models/weights.best.basic_mlp.hdf5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess audio and extract MFCC features
def preprocess_audio(audio_path):
    try:
        # Load audio using librosa
        audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast') 
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Calculate mean of MFCCs
        
        # You can perform additional preprocessing steps or feature scaling here
        
        return mfccs_mean  # Return preprocessed audio features
    
    except Exception as e:
        print("Error encountered while processing audio:", e)
        return None

# Function to make audio predictions
def predict_sound(audio):
    # Preprocess the audio
    processed_audio = preprocess_audio(audio)

    if processed_audio is not None:
        # Make prediction
        prediction = model.predict(np.expand_dims(processed_audio, axis=0))
        return prediction
    else:
        return None

# Streamlit app
st.title('Predict Sound from Audio')
st.write('Upload an audio file to predict its type!')

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    # Display the uploaded file info
    st.write('Uploaded Audio File:', uploaded_file.name)

    # Make a prediction
    prediction = predict_sound(uploaded_file)
    if prediction is not None:
        class_names = [
            'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
        ]
        predicted_class = class_names[np.argmax(prediction)]

        st.write(f"Predicted Sound: {predicted_class}")
    else:
        st.write("Error processing audio. Please upload a valid WAV file.")

>>>>>>> eb5a706b74444f0f006757b4ae7e10e5087147bd
