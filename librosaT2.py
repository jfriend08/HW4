import librosa
import numpy as np

import sklearn
import sklearn.cluster
import sklearn.pipeline

import matplotlib.pyplot as plt
import seaborn
seaborn.set(style='ticks')


# We'll build the feature pipeline object here

# First stage is a mel-frequency specrogram of bounded range
MelSpec = librosa.util.FeatureExtractor(librosa.feature.melspectrogram, 
                                        n_fft=2048,
                                        n_mels=128,
                                        fmax=librosa.midi_to_hz(116), 
                                        fmin=librosa.midi_to_hz(24))

# Second stage is log-amplitude; power is relative to peak in the signal
LogAmp = librosa.util.FeatureExtractor(librosa.logamplitude, 
                                       ref_power=np.max)


# Third stage transposes the data so that frames become samples
Transpose = librosa.util.FeatureExtractor(np.transpose)

# Last stage stacks all samples together into one matrix for training
Stack = librosa.util.FeatureExtractor(np.vstack, iterate=False)

# Now, build a learning object.  We'll use mini-batch k-means with default parameters.
C = sklearn.cluster.MiniBatchKMeans()

# Now, chain them all together into a pipeline
ClusterPipe = sklearn.pipeline.Pipeline([('Mel spectrogram', MelSpec), 
                                         ('Log amplitude', LogAmp),
                                         ('Transpose', Transpose),
                                         ('Stack', Stack),
                                         ('Cluster', C)])


# Let's build a model using just the first 20 seconds of the example track
y_train, sr = librosa.load(librosa.util.example_audio_file(), duration=20, offset=0.0)


# Fit the model.
# [y_train] will be passed through the entire feature pipeline before k-means is trained
ClusterPipe.fit([y_train])


# We can plot the resulting centroids
plt.figure(figsize=(4, 4))

librosa.display.specshow(C.cluster_centers_.T)

plt.xticks(range(len(C.cluster_centers_)))
plt.xlabel('Cluster #')

plt.ylabel('Mel frequency')

plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
plt.show()

# Now we can make predictions, in this case, frame-level cluster identifiers.
# Let's run it on the training data, just to be sure it worked.
print "ClusterPipe.predict([y_train])", ClusterPipe.predict([y_train])

# Now we can test it on a different portion of the track: [20s, 25s]
y_test, sr = librosa.load(librosa.util.example_audio_file(), duration=5, offset=20.0)
print "ClusterPipe.predict([y_test])", ClusterPipe.predict([y_test])