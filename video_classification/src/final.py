from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request
import librosa
import sounddevice
import librosa
from pathlib import Path
from scipy.io.wavfile import write
import numpy as np # linear algebra
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import numpy
import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
lb=LabelEncoder()
import warnings
from sklearn.preprocessing import LabelEncoder,StandardScaler
# Your existing code here
import tensorflow as tf
model1 = tf.keras.models.load_model('./res_model.h5')

# Initialize Flask application
app = Flask(__name__)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}


#audio part
processed_data_path='./processed_data.csv'
df=pd.read_csv(processed_data_path)
X=df.drop(labels='Emotion',axis=1)
Y=df['Emotion']
lb=LabelEncoder()
Y=np_utils.to_categorical(lb.fit_transform(Y))
numpy.save('classes.npy', lb.classes_)
fs=44100
second=2.5


def add_noise(data,random=False,rate=0.035,threshold=0.075):
    if random:
        rate=np.random.random()*threshold
    noise=rate*np.random.uniform()*np.amax(data)
    augmented_data=data+noise*np.random.normal(size=data.shape[0])
    return augmented_data

def shifting(data,rate=1000):
    augmented_data=int(np.random.uniform(low=-5,high=5)*rate)
    augmented_data=np.roll(data,augmented_data)
    return augmented_data

def pitching(data, sr, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    n_steps = np.log2(pitch_factor) * 12
    return librosa.effects.pitch_shift(data, n_steps=n_steps, sr=sr)

def streching(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)
    
    

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))
def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=2.5,offset=0.6)
    aud=extract_features(data,sr)
    audio=np.array(aud)
    
    noised_audio=add_noise(data,random=True)
    aud2=extract_features(noised_audio,sr)
    audio=np.vstack((audio,aud2))
    
    pitched_audio=pitching(data,sr,random=True)
    aud3=extract_features(pitched_audio,sr)
    audio=np.vstack((audio,aud3))
    
    pitched_audio1=pitching(data,sr,random=True)
    pitched_noised_audio=add_noise(pitched_audio1,random=True)
    aud4=extract_features(pitched_noised_audio,sr)
    audio=np.vstack((audio,aud4))
    
    return audio




# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    # Check if the file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Save the uploaded file to a location
        file_path = './audio/' + filename
        file.save(file_path)

        # Perform emotion prediction using the uploaded file
        emotion = predict_emotion(file_path)

        # Return the predicted emotion
        return 'Predicted Emotion: ' + emotion

    return 'Invalid file', 400


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# Load the haarcascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# Function to detect and predict emotions from webcam feed
def detect_emotions():
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = model.predict(roi_gray)[0]
            maxindex = int(np.argmax(prediction))
            emotion_label = emotion_dict[maxindex]

            cv2.putText(frame, emotion_label, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/audio')
def index1():
    return render_template('audio.html')
# Route for the home page


@app.route('/video')
def index2():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    while(1):
        record_voice = sounddevice.rec(int(second * fs), samplerate=fs, channels=2)
        sounddevice.wait()
        write(Path('./audio/', '{}.wav'.format(str(0))), fs, record_voice)
        data, sr = librosa.load(Path('./audio/', '{}.wav'.format(str(0))))

        features = extract_features(data, sr)
        X, Y = [], []
        for i in features:
            X.append(i)
        cols=2376-len(X)
        X.extend([None] * cols)
        extract=pd.DataFrame(X)
        extract = extract.values.reshape(1, -1)
        extract=pd.DataFrame(extract)
        extract=extract.fillna(0)
        ext=extract.values


        scaler=StandardScaler()
        Input=scaler.fit_transform(ext)

        y_pr2 = model1.predict(ext)
        y_pr2 = np.argmax(y_pr2, axis=1)
        original_label = lb.inverse_transform(y_pr2)
        print(original_label)
    
    # Render the prediction result template
    return render_template('result.html', prediction=original_label[0])




@app.route('/')
def index3():
    return render_template('home.html')


@app.route('/contact')
def index4():
    return render_template('contact.html')


# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(detect_emotions(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
