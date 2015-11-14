from sklearn.datasets import load_iris
from  scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import random, sys
import matplotlib.pyplot as plt
from scipy import ndimage

import cPickle as pickle
import scipy.io.wavfile as wav
from pylab import plt
from scipy.signal import butter, lfilter, freqz
from sklearn.cross_validation import train_test_split
from librosa.util import normalize

class mykmeans(object):
  def __init__(self, num_cluster, max_iter=50):
    self.Centroids = []
    self.NUM_CLUSTER = num_cluster
    self.MAX_ITER = max_iter
    print "init"

  def centroids(self):
    return self.Centroids

  def genRandPoints(self, points, k):
    #return k random number of points from points
    randomPoints = []
    random.seed(19850920)
    for i in xrange(k):
      idx = random.randint(0,len(points)-1)
      point = points[idx]
      randomPoints.append(point)
    return randomPoints

  def getGroup(self, points, centroids=None):
    groupResult = []
    if centroids == None:
        centroids = self.Centroids
    elif len(centroids) == 0:
      return

    for point in points:
      min_dis = sys.float_info.max
      min_groupIdx = -1
      for center_idx in xrange(len(centroids)):
        # print "centroids[center_idx]", centroids[center_idx]
        dist = euclidean(point, centroids[center_idx])
        if dist < min_dis:
            min_dis = dist
            min_groupIdx = center_idx
      groupResult.append(min_groupIdx)

    return groupResult

  def getCentroids(self, points_inGroup):
    newCentroids = []
    for eachGroup in points_inGroup:
      if len(eachGroup) == 0:
        continue
      newCentroids.append(np.sum(eachGroup, axis=0)/len(eachGroup))
    return newCentroids

  def mykmeans(self, points):
    k = self.NUM_CLUSTER
    max_iter = self.MAX_ITER
    centers = self.genRandPoints(points, k)
    print "Start kmeans"
    iterationCount = 1
    while iterationCount <= max_iter:
      # print "Iteration:", iterationCount
      # print "centers:", centers
      points_inGroup = [[]for i in xrange(len(centers))]
      dist_total = 0
      for point_idx in xrange(len(points)):
        min_dis = sys.float_info.max
        min_groupIdx = -1
        for center_idx in xrange(len(centers)):
          dist = euclidean(points[point_idx], centers[center_idx])
          if dist < min_dis:
            min_dis = dist
            min_groupIdx = center_idx
        points_inGroup[min_groupIdx].append(points[point_idx])
        dist_total += min_dis
      centers = self.getCentroids(points_inGroup)
      # print "dist_total", dist_total
      iterationCount +=1
    self.Centroids = centers

class mykmeans_multi(mykmeans):
  def __init__(self, num_cluster, max_iter):
    self.Centroids = []
    self.distortion = []
    self.MAX_ITER = max_iter
    self.NUM_CLUSTER = num_cluster

  def getGroup(self, points, centroids=None):
    groupResult = []
    if centroids == None:
        centroids = self.getBestCentroids()
    elif len(centroids) == 0:
      return
    # print "Kmeans multi len(centroids)", len(centroids)
    for point in points:
      min_dis = sys.float_info.max
      min_groupIdx = -1
      for center_idx in xrange(len(centroids)):
        dist = euclidean(point, centroids[center_idx])
        if dist < min_dis:
            min_dis = dist
            min_groupIdx = center_idx
      groupResult.append(min_groupIdx)

    return groupResult

  def printHey(self):
    print "hey"
    print "self.distortion", self.distortion

  def getBestCentroids(self):
    return sorted(self.Centroids, key=lambda tup: tup[0])[0][1]

  def getDistortionsOverIteration(self):
    return self.distortion

  def getCentroidsOverIteration(self):
    centroids_all = []
    for (dist_total, centers) in self.Centroids:
      centroids_all.append(centers)
    return centroids_all

  def mykmeans(self, points):
    max_iter = self.MAX_ITER
    k = self.NUM_CLUSTER
    centers = self.genRandPoints(points, k)
    iterationCount = 1
    noImproveCount = 0
    print "Start kmeans"
    # print "np.array(centers).shape", np.array(centers).shape
    # print "centers", centers
    while iterationCount <= max_iter and noImproveCount < 100:
      # print "Iteration:", iterationCount, "noImproveCount", noImproveCount
      points_inGroup = [[]for i in xrange(len(centers))]
      dist_total = 0
      for point_idx in xrange(len(points)):
        min_dis = sys.float_info.max
        min_groupIdx = -1
        for center_idx in xrange(len(centers)):
          dist = euclidean(points[point_idx], centers[center_idx])
          if dist < min_dis:
            min_dis = dist
            min_groupIdx = center_idx
        points_inGroup[min_groupIdx].append(points[point_idx])
        dist_total += min_dis
      self.Centroids.append((dist_total, centers))
      if len(self.distortion)>0:
        improvement = abs(self.distortion[-1] - dist_total)/self.distortion[-1]
        # print "improvement", improvement
      if len(self.distortion)>0 and  abs(self.distortion[-1] - dist_total)/self.distortion[-1] < 0.0001:
        noImproveCount+=1
      else:
        noImproveCount=0
      self.distortion.append(dist_total)
      centers = self.getCentroids(points_inGroup)
      iterationCount +=1

class mykmeans_pp(mykmeans_multi):
  def genRandPoints(self, points, k):
    randomPoints = []
    random.seed(19850920)
    np.random.seed(1234567890)
    idx = random.randint(0,len(points)-1)
    point = points[idx]
    randomPoints.append(point)
    # print "point", point
    # print "self.MAX_ITER", self.MAX_ITER, "self.NUM_CLUSTER", self.NUM_CLUSTER
    while len(randomPoints)<self.NUM_CLUSTER:
      probabilityDist = []
      for point in points:
        min_dis = sys.float_info.max
        shortestCenter_idx = -1
        for center_idx in xrange(len(randomPoints)):
          dist = euclidean(point, randomPoints[center_idx])
          if dist < min_dis:
            min_dis = dist
            shortestCenter_idx = center_idx
        probabilityDist.append(min_dis)
      # print "probabilityDist", probabilityDist
      probabilityDist = np.array(probabilityDist)/sum(probabilityDist)
      # print "probabilityDist", probabilityDist
      idx = np.random.choice(len(points), 1, p=probabilityDist)
      point = points[idx]
      # print "next point", point
      randomPoints.extend(point)
    # print "len(randomPoints)", len(randomPoints)
    return randomPoints
  # def mykmeans(self, X):
  #   super(mykmeans_pp, self).mykmeans(X)

def plotPCA(X, y, centroids):
  pca = PCA()
  pca.fit(X)
  X_pca = pca.transform(X)
  centroids_pca = pca.transform(centroids)
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
  plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], linewidths=0, s=200, marker=(5, 1), c=[i for i in xrange(len(centroids_pca))])
  plt.xlabel("first principal component")
  plt.ylabel("second principal component")
  plt.show()
  plt.legend()

def plotPCA_multi(X, y, centroids, centroids_all, distortions):
  pca = PCA()
  pca.fit(X)
  X_pca = pca.transform(X)
  centroids_pca = pca.transform(centroids)
  # centroids_all_pca = pca.transform(centroids_all)
  centroids_all_pca = map(pca.transform, np.array(centroids_all))
  plt.subplot(311)
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
  plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], linewidths=0, s=200, marker=(5, 1), c=[i for i in xrange(len(centroids_pca))])
  plt.xlabel("first principal component")
  plt.ylabel("second principal component")
  plt.xlim(-4, 4)
  plt.ylim(-3, 3)
  plt.subplot(312)
  for cen in centroids_all_pca:
    plt.scatter(cen[:, 0], cen[:, 1], linewidths=0, s=50, marker=(5, 1), c=[i for i in xrange(len(cen))])
  plt.xlabel("first principal component")
  plt.ylabel("second principal component")
  plt.xlim(-4, 4)
  plt.ylim(-3, 3)
  plt.subplot(313)
  plt.plot(distortions)
  plt.show()
  # plt.legend()

def main():
  #Data import and preprocessing
  scaler = StandardScaler()
  iris = load_iris()
  X, y = iris.data, iris.target
  scaler.fit(X)
  X = scaler.transform(X)

  #regular kmeans
  kmeans = mykmeans(10, 50)
  kmeans.mykmeans(X)
  centroids = kmeans.centroids()
  y = kmeans.getGroup(X)
  plotPCA(X, y, centroids)

  #kmeans_multi
  mykmeans_multi = mykmeans_multi(10, 100)
  mykmeans_multi.mykmeans(X)
  centroids = mykmeans_multi.getBestCentroids()
  y = mykmeans_multi.getGroup(X, centroids)
  centroids_all = mykmeans_multi.getCentroidsOverIteration()
  distortions = mykmeans_multi.getDistortionsOverIteration()
  print "distortions", distortions
  plotPCA_multi(X, y, centroids, centroids_all, distortions)

  #kmeans_pp
  mykmeans_pp = mykmeans_pp(10, 100)
  mykmeans_pp.mykmeans(X)
  centroids = mykmeans_pp.getBestCentroids()
  y = mykmeans_pp.getGroup(X, centroids)
  centroids_all = mykmeans_pp.getCentroidsOverIteration()
  distortions = mykmeans_pp.getDistortionsOverIteration()
  print "distortions", distortions
  plotPCA_multi(X, y, centroids, centroids_all, distortions)

# scaler = StandardScaler()
# iris = load_iris()
# X, y = iris.data, iris.target
# scaler.fit(X)
# X = scaler.transform(X)

# #kmeans_multi
# mykmeans_multi = mykmeans_multi(10, 100)
# mykmeans_multi.mykmeans(X)
# centroids = mykmeans_multi.getBestCentroids()
# y = mykmeans_multi.getGroup(X, centroids)
# centroids_all = mykmeans_multi.getCentroidsOverIteration()
# distortions = mykmeans_multi.getDistortionsOverIteration()
# print "distortions", distortions
# plotPCA_multi(X, y, centroids, centroids_all, distortions)

# mykmeans_pp = mykmeans_pp(10, 100)
# mykmeans_pp.mykmeans(X)
# centroids = mykmeans_pp.getBestCentroids()
# y = mykmeans_pp.getGroup(X, centroids)
# centroids_all = mykmeans_pp.getCentroidsOverIteration()
# distortions = mykmeans_pp.getDistortionsOverIteration()
# print "distortions", distortions
# plotPCA_multi(X, y, centroids, centroids_all, distortions)

