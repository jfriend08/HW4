# Feature extraction example
# Beat tracking example

import librosa
import numpy as np

import sklearn
import sklearn.cluster
import sklearn.pipeline
from sklearn.cluster import MiniBatchKMeans

import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks')

import numpy as np
import librosa

# Load the example clip
y, sr = librosa.load(librosa.util.example_audio_file(), duration=20, offset=0.0)
print "y.shape", y.shape
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
print "S.shape", S.shape
librosa.display.specshow(librosa.logamplitude(S,ref_power=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

print "S.shape", S.shape
S_log = librosa.logamplitude(S,ref_power=np.max)
print "S_log.shape", S_log.shape
S_log_processed = np.vstack(np.transpose(S_log))
print "S_log_processed.shape", S_log_processed.shape
print S_log_processed
clf = MiniBatchKMeans().fit(S_log_processed) #X : array-like, shape = [n_samples, n_features]

print clf.predict(S_log_processed)

y_test, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=20.0)
print "y_test.shape", y_test.shape
clf.predict([y_test])

