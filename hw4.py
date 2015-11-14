from sklearn.datasets import load_iris
from scipy.spatial import distance
import random
import numpy as np
import matplotlib.pyplot as plt

class Node(object):
  def __init__(self, coords, label=-1):
    self.pos = coords
    self.label = label

  def __cmp__(self, other):
    if len(self.pos) != len(other.pos):
      return False

    for i in xrange(len(self.pos)):
      if 0.000000001 < abs(self.pos[i] - other.pos[i]):
        return False

    return True

  def __add__(self, other):
    if isinstance(other, Node):
      return self.pos + other.pos
    else:
      return self.pos + other

class Cluster(object):
  def __init__(self, centroid):
    self.centroid = centroid
    self.groups = set()
    self.newGroups = set()

  def addNode(self, node):
    self.newGroups.add(node)

  def update(self):
    '''
    rtype: Boolean, whether centroid is moved
    '''
    new_coord = reduce(lambda x, y: y + x, self.newGroups) / len(self.newGroups)
    newCentroid = Node(new_coord)

    prev_dists = sum(map(lambda n: distance.euclidean(self.centroid.pos, n.pos), self.groups))
    now_dists = sum(map(lambda n: distance.euclidean(newCentroid.pos, n.pos), self.newGroups))

    self.centroid = newCentroid
    self.groups = self.newGroups
    self.newGroups = set()

    if prev_dists > 0 and now_dists > 0 and prev_dists / now_dists < 0.0001:
      return True
    else:
      return False

  @property
  def num(self):
    return len(self.groups)

def plotGroups(groups):
  numGroups = len(groups)

  for i in xrange(numGroups):
    dots = np.asarray(map(lambda n: n.pos, groups[i].groups))
    plt.plot(dots[:, :1], dots[:, 1:2])

  plt.show()

def mykmeans(data, k, max_iter=50):
  init_centroids = random.sample(data, k)
  clusters = map(lambda node: Cluster(node), init_centroids)

  for it in xrange(max_iter):
    # grouping
    for i in xrange(len(data)):
      idx = np.argmin(map(lambda cluster: distance.euclidean(cluster.centroid.pos, data[i].pos), clusters))
      clusters[idx].addNode(data[i])

    # Check if convergence
    isConvergence = map(lambda cluster: cluster.update(), clusters)
    if not False in isConvergence:
      break

  return clusters

def mykmeans_multi(data, k, max_iter=50, runs=100):
  for i in xrange(runs):
    groups = mykmeans(data, k, max_iter)

def mykmeans_plus(data, k, max_iter=50):
  pass

def main():
  random.seed(5181986)
  iris = load_iris()
  X, y = iris.data, iris.target
  data = map(lambda pair: Node(pair[0], pair[1]), zip(X, y))

  groups = mykmeans(data, 3)
  plotGroups(groups)

if __name__ == '__main__':
  main()
