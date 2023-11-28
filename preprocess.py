import librosa
import librosa.display
import os
import pandas as pd

path = "/content/drive/MyDrive/RAVDESS"
random_seed = 41
emotions = {'01' : 'Neutral', '02' : 'Calm', '03' : 'Happy',  '04' : 'Sad',
              '05' : 'Angry', '06' : 'Fearful', '07' : 'Disgust', '08' : 'Surprised'}
level = {'01' : 'Normal', '02' : 'Strong'}
statement = {'01' : 'Kids are talking by the door' , '02' : 'Dogs are sitting by the door'}

def GenderClassifier(x):
  x = int(x)
  if (x % 2 == 0):
    return "Female"
  return "Male"

class Preprocess():
  def __init__(self, random_seed, dir):
    self.seed = random_seed
    self.dir = dir

    self.emotion = []
    self.id = []
    self.intensity = []
    self.sentence = []
    self.path = []

  def load(self):
      for dirname, _, filenames in os.walk(self.dir):
        for filename in filenames:
          self.path.append(os.path.join(dirname, filename))
          Sequence = filename.split('-')
          self.emotion.append(Sequence[2])
          self.intensity.append(Sequence[3])
          self.sentence.append(Sequence[4])
          self.append(Sequence[6].split('.')[0])
  
  def topd(self):
    df = pd.DataFrame({'ID' : self.id, 'Emotion' : self.emotion, 
                       'Intensity' : self.intensity, 'Statement' : self.sentence, 'Path' : self.path})
    df['Emotion'] = df['Emotion'].map(emotions)
    df['Intensity'] = df['Intensity'].map(level)
    df['Statement'] = df['Statement'].map(statement)
    df['Gender'] = df['ID'].apply(lambda x : GenderClassifier(x))
    return df




if __name__=='__main__':
  PreProcess(41, 
