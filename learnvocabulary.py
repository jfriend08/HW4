from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from  scipy.spatial.distance import euclidean
import os, sys
import numpy as np
from scipy.io import wavfile
import cPickle as pickle
import scipy.io.wavfile as wav
import librosa
from pylab import plt
from scipy.signal import butter, lfilter, freqz
from sklearn.cross_validation import train_test_split
from librosa.util import normalize
import kmeans as km
import cProfile

def getData(path, clipWindow=60000):
  samples = pickle.load( open( path, "rb" ) )
  X = []
  y = []
  for genere_idx in xrange(len(samples.keys())):
    genere = samples.keys()[genere_idx]
    songs = samples[genere]
    for song_idx in xrange(len(songs)):
      song = songs[song_idx]
      song = [clip[:clipWindow] for clip in song] #some clip has different num of signal
      X.append(song)
      y.append(genere_idx)
  print "X size/number of songs:", len(X)
  print "Number of clips per song:", len(X[0])
  print "y size:", len(y)
  return X, y

def flaten(X):
  return [val for sublist in X for val in sublist]

def plus1(X):
  return X+1

def MFCC(signal, sr=22050):
  # S = librosa.feature.melspectrogram(y=np.array(signal), sr=sr, n_mels=128,fmax=8000)
  # return librosa.logamplitude(S,ref_power=np.max)
  return librosa.feature.mfcc(y=np.array(signal), sr=sr, n_mfcc=12)

def learnvocabulary(X, k, method="kpp", Iter=100, transpose=False):
  X = np.array([[MFCC(clip) for clip in song] for song in X])
  # X = [[MFCC(clip) for clip in song] for song in X]
  print "After MFCC X.shape", X.shape
  X_train_flattened = [val for sublist in X for val in sublist]
  print "X_train_flattened.shape", np.array(X_train_flattened).shape
  librosa.display.specshow(X_train_flattened[0], x_axis='time')
  plt.colorbar()
  plt.title('MFCC X_train_flattened[0]')
  plt.tight_layout()
  plt.show()

  if transpose:
    X_train_flattened = np.array(map(np.transpose, X_train_flattened))
    print "After transpose X_train_flattened.shape", X_train_flattened.shape

  X_train_flattened_norm = normalize(X_train_flattened, norm=2)
  X_train_flattened_norm_final = np.array([mfcc for clip in X_train_flattened_norm for mfcc in clip])
  print "X_train_flattened_norm_final.shape", X_train_flattened_norm_final.shape
  if method == "kpp":
    kmeans = km.mykmeans_pp(k, Iter) #numCluster, numIter
  elif method == "multi":
    kmeans = km.mykmeans_multi(k, Iter) #numCluster, numIter
  else:
    print "no such method"
  kmeans.mykmeans(X_train_flattened_norm_final)
  centroids = kmeans.getBestCentroids()
  centroids_all = kmeans.getCentroidsOverIteration()
  distortions = kmeans.getDistortionsOverIteration()
  print "Last distortion:", distortions[-1]
  y = kmeans.getGroup(X_train_flattened_norm_final)
  km.plotPCA(X_train_flattened_norm_final, y, centroids)
  km.plotPCA_multi(X_train_flattened_norm_final, y, centroids, centroids_all, distortions)
  return centroids

def getbof(X, y, centroids, transpose=False):
  finalResult = []
  X = np.array([MFCC(clip) for song in X for clip in song])
  if transpose:
    X = np.array(map(np.transpose, X))
    print "After transpose X.shape", X.shape

  y_flatten = []
  for genre in y:
    tmp = [genre for i in xrange(len(X)/len(y))]
    y_flatten.extend(tmp)

  print "getbof X.shape", X.shape
  for song_idx in xrange(len(X)):
    song = X[song_idx]
    centroidCount = [0 for i in xrange(len(centroids))]
    for mfcc in song:
      min_dis = sys.float_info.max
      shortestCenter_idx = -1
      for center_idx in xrange(len(centroids)):
        dist = euclidean(mfcc, centroids[center_idx])
        if dist < min_dis:
          min_dis = dist
          shortestCenter_idx = center_idx
      centroidCount[shortestCenter_idx] += 1

    # for c_idx in xrange(len(centroidCount)):
    #   count = centroidCount[c_idx]
    #   centroidCount[c_idx] = [(song_idx,c_idx), count]
      # if count != 0:
      #   centroidCount[c_idx] = [(song_idx,c_idx), count]
      # else:
      #   centroidCount[c_idx] = [0]
    finalResult.append(centroidCount)
  return np.array(finalResult), np.array(y_flatten)

# # X1, y1 = getData("./data/data_small5.in")
# X, y = getData("./data/data_small8II.in")

# # print "np.array(X1).shape", np.array(X1).shape
# # print "np.array(X).shape", np.array(X).shape

# # X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.25, random_state=2010)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2010)
# # print 'np.array(X1_train).shape', np.array(X1_train).shape
# print 'np.array(X_train).shape', np.array(X_train).shape
# # centroids = learnvocabulary(X1_train, 20, "kpp", 50, False) #X, k, method, num of Iter, transpose mfcc
# centroids = learnvocabulary(X_train, 20, "kpp", 50, False) #X, k, method, num of Iter, transpose mfcc

# '''Important Part'''
# X, y = getData()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2010)
# centroids = learnvocabulary(X_train)
# X_train_bofCounts, y_train = getbof(X_train, y_train, centroids)

# print "X_train_bofCounts.shape", X_train_bofCounts.shape
# print "y_train", y_train

# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_bofCounts)
# print "X_train_tfidf.shape", X_train_tfidf.shape
# y_train = np.array(y_train)
# print "y_train.shape", y_train.shape

# #Training a classifier
# from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_tfidf, y_train)
# print "clf", clf

# #Predict
# X_test_bofCounts, y_test = getbof(X_test, y_test, centroids)
# X_test_tfidf = tfidf_transformer.fit_transform(X_test_bofCounts)
# predicted = clf.predict(X_test_tfidf)
# print "naive bayes", np.mean(predicted == y_test)


# ''' import data '''
# X, y = getData()
# print "before MFCC", X[0][0][:100]
# print "before MFCC X[0][0].shape", X[0][0].shape

# ''' do MFCC for clips from all songs '''
# X = [[MFCC(clip) for clip in song] for song in X]
# print "after MFCC", X[0][0][:100]
# print "after MFCC X[0][0].shape", X[0][0].shape

# ''' split data '''
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2010)
# print "X_train size:", len(X_train)
# X_train_flattened = flaten(X_train)
# print "X_train_flattened size:", len(X_train_flattened)

# librosa.display.specshow(X_train_flattened[0], x_axis='time')
# plt.colorbar()
# plt.title('MFCC X_train_flattened[0]')
# plt.tight_layout()
# plt.show()

# print "befor norm", X_train_flattened[0][0]
# X_train_flattened_norm = normalize(X_train_flattened, norm=2)

# print "X_train_flattened_norm.shape", X_train_flattened_norm.shape
# print "after norm", X_train_flattened_norm[0][0]

# X_train_flattened_norm_final = np.array([mfcc for clip in X_train_flattened_norm for mfcc in clip])
# print "X_train_flattened_norm_final.shape", np.array(X_train_flattened_norm_final).shape

# mykmeans_pp = km.mykmeans_pp(10, 100)
# mykmeans_pp.mykmeans(X_train_flattened_norm_final)
# centroids = mykmeans_pp.getBestCentroids()
# # print "centroids", centroids
# y_kmeans = mykmeans_pp.getGroup(X_train_flattened_norm_final, centroids)
# centroids_all = mykmeans_pp.getCentroidsOverIteration()
# distortions = mykmeans_pp.getDistortionsOverIteration()
# print "distortions", distortions
# km.plotPCA(X_train_flattened_norm_final, y_kmeans, centroids)
# km.plotPCA_multi(X_train_flattened_norm_final, y, centroids, centroids_all, distortions)


# mykmeans_multi = km.mykmeans_multi(10, 100)
# mykmeans_multi.mykmeans(X_train_flattened_norm_final)
# centroids = mykmeans_multi.getBestCentroids()
# # print "centroids", centroids
# y_kmeans = mykmeans_multi.getGroup(X_train_flattened_norm_final, centroids)
# centroids_all = mykmeans_multi.getCentroidsOverIteration()
# distortions = mykmeans_multi.getDistortionsOverIteration()
# print "distortions", distortions
# km.plotPCA(X_train_flattened_norm_final, y_kmeans, centroids)
# km.plotPCA_multi(X_train_flattened_norm_final, y_kmeans, centroids, centroids_all, distortions)

# kmeans = km.mykmeans(10, 50)
# kmeans.mykmeans(X_train_flattened_norm_final)
# centroids = kmeans.centroids()
# y = kmeans.getGroup(X_train_flattened_norm_final)
# km.plotPCA(X_train_flattened_norm_final, y, centroids)



# #Data import and preprocessing
# scaler = StandardScaler()
# iris = load_iris()
# X, y = iris.data, iris.target
# scaler.fit(X)
# X = scaler.transform(X)

# # # #regular kmeans
# # # kmeans = km.mykmeans(10, 50)
# # # kmeans.mykmeans(X)
# # # centroids = kmeans.centroids()
# # # y = kmeans.getGroup(X)
# # # km.plotPCA(X, y, centroids)

# #kmeans_pp
# mykmeans_pp = km.mykmeans_pp(10, 100)
# mykmeans_pp.mykmeans(X)
# centroids = mykmeans_pp.getBestCentroids()
# y = mykmeans_pp.getGroup(X, centroids)
# centroids_all = mykmeans_pp.getCentroidsOverIteration()
# distortions = mykmeans_pp.getDistortionsOverIteration()
# print "distortions", distortions
# km.plotPCA_multi(X, y, centroids, centroids_all, distortions)
