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
model = tf.keras.models.load_model('./res_model.h5')
app = Flask(__name__)


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




# Route for the home page
@app.route('/')
def home():
    return render_template('audio.html')



# Route to handle the audio upload and prediction
@app.route('/predict', methods=['POST'])
def predict():

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

    y_pr2 = model.predict(ext)
    y_pr2 = np.argmax(y_pr2, axis=1)
    original_label = lb.inverse_transform(y_pr2)
    print(original_label)
    return render_template('result.html', prediction=original_label[0])


if __name__ == '__main__':
    app.run()