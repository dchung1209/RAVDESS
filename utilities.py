import librosa
import matplotlib.plt as plt
import IPython.display as ipd
import numpy as np
import pandas as pd

class RAVDESS():
    emotions = {'01' : 'Neutral', '02' : 'Calm', '03' : 'Happy',  '04' : 'Sad',
              '05' : 'Angry', '06' : 'Fearful', '07' : 'Disgust', '08' : 'Surprised'}

    level = {'01' : 'Normal', '02' : 'Strong'}

    statement = {'01' : 'Kids are talking by the door' , '02' : 'Dogs are sitting by the door'}

    def __init__(self):
      self.invert_emotions = dict(zip(self.emotions.values(), self.emotions.keys()))
      
    def GenderClassifier(self, x):
        x = int(x)
        if (x % 2 == 0):
            return "Female"
        else:
            return "Male"
    
    def load(self, root):
        emotion = []
        intensity = []
        path = []
        sentence = []
        id = []

        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                path.append(os.path.join(dirname, filename))
                Sequence = filename.split('-')
                emotion.append(Sequence[2])
                intensity.append(Sequence[3])
                sentence.append(Sequence[4])
                id.append(Sequence[6].split('.')[0])
        
        df = pd.DataFrame({
            'ID' : id,
            'Emotion' : emotion,
            'Intensity' : intensity,
            'Statement' : sentence,
            'Path' : path
            })

        df['Emotion'] = df['Emotion'].map(emotions)
        df['Intensity'] = df['Intensity'].map(level)
        df['Statement'] = df['Statement'].map(statement)
        df['Gender'] = df['ID'].apply(lambda x : self.GenderClassifier(x))
        return df


class AudioUtils(RAVDESS):
    def __init__(self, path):
        self.path = path

    def MelSpec(self, root, channel_num = 1):
        y, sr = librosa.load(root, sr = 48000)
        S = librosa.feature.melspectrogram(y = y, n_fft = 2048, hop_length = 480, sr = sr, win_length = 1920)
        logS = librosa.power_to_db(S, ref=np.max)

        if (channel_num != 1):
            logS_delta = librosa.feature.delta(logS)
            logS_ddelta = librosa.feature.delta(logS_delta)
            return  np.stack([logS, logS_delta, logS_ddelta], axis=-1)
        
        return logS

    def MelSpecDraw(self, s):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(s, x_axis='time', y_axis='mel', sr = 48000)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')
        plt.show()

    def WavePlotDraw(self):
        y, sr = librosa.load(self.path, sr = 48000)
        librosa.display.waveshow(y = y, sr = sr)


    def HearAudio(self):
        audio, sr = librosa.load(self.path, sr = 48000)
        ipd.display(ipd.Audio(data=audio, rate=sr))


    