
from flask import Flask, render_template, request
import os
import numpy as np
from audio_utils import preprocess_audio
from tensorflow.keras.models import load_model

app = Flask(__name__)


model = load_model('model.h5')
n_mfcc = 40
max_length = 209809


def predict_emotion(file_path):
    processed_data = preprocess_audio(file_path, n_mfcc, max_length)
    prediction = model.predict(processed_data)
    emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    return predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        audio_file = request.files['audio']
        file_path = 'uploads/' + audio_file.filename
        audio_file.save(file_path)
        predicted_emotion = predict_emotion(file_path)
        os.remove(file_path)  # Remove the uploaded file
        return render_template('index.html', predicted_emotion=predicted_emotion)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
