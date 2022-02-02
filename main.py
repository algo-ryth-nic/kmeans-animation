import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def initializeCentroids(k: int, points: np.ndarray) -> np.ndarray:
  '''
  Randomly selects k points as inital centroids locations
  k: no. of clusters
  points: n x d, where is n the number of points & d is no. of dimensions (d=2 for now)
  '''
  indices = []
  n = points.shape[0]

  if k > len(points):
    print("Error: k is greater than the number of points")
    return
  else:
    while len(indices) < k:
      index = np.random.randint(low= 0, high=n)
      if not index in indices:
        indices.append(index)
    return points[indices, :] 

def assignPointsToClusters(centroids: np.ndarray, points: np.ndarray) -> np.ndarray:
  '''
  Assigns each point to the closest centroid
  centroids: k x d, where is k the number of clusters & d is no. of dimensions (d=2 for now)
  points: n x d, where is n the number of points & d is no. of dimensions (d=2 for now)
  
  RETURNS clusters: n x 1, where is n the number of points
  '''
  n = points.shape[0] # no. of points
  k = centroids.shape[0] # no. of clusters``
  clusters = np.zeros(shape=(n,)) # n x 1, where n is the number of points

  # for each point, finds the distance to each centroid
  for i in range(n):
    distances = []
    for j in range(k):
      # euclidean distance
      distances.append(np.linalg.norm(points[i, :] - centroids[j, :]))
    
    clusters[i] = distances.index(min(distances))
  return clusters

def optimizeCentroids(centroids: np.ndarray, points: np.ndarray, clusters: np.ndarray) -> np.ndarray:
  '''
  Updates the centroid locations to be the mean of the points assigned to the cluster
  centroids: k x d, where is k the number of clusters & d is no. of dimensions (d=2 for now)
  points: n x d, where is n the number of points & d is no. of dimensions (d=2 for now)
  clusters: 
  '''
  n = points.shape[0] # no. of points
  k = centroids.shape[0] # no. of clusters
  newCentroids = np.zeros(shape=(k, points.shape[1])) # k x d, where k is the number of clusters & d is no. of dimensions (d=2 for now)

  # for each cluster  
  for i in range(k):
    # get the points assigned to that cluster
    pointsInCluster = points[clusters == i]
    newCentroids[i, :] = np.mean(pointsInCluster, axis=0)
  return newCentroids


class KmeansAnimate():
  def __init__(self, k: int, data: np.ndarray, startCentroids=None):
    self.k = k
    self.points = data
    self.initialCentroids = startCentroids

  def _generatorFuncAnimate(self):
    '''
    Animates the kmeans algorithm
    points: n x d, where is n the number of points & d is no. of dimensions 
    centroids: k x d, where is k the number of clusters & d is no. of dimensions 
    clusters: assigned to clusters to points
    initialCentroids (optional) = k x d, where is k the number of clusters & d is no. of dimensions 
    '''

    if self.initialCentroids is None:
      centroids = initializeCentroids(self.k, self.points)
    elif self.initialCentroids.shape[0] != self.k:
      RuntimeError('initialCentroids shape != k')
    else:
      centroids = self.initialCentroids

    clusters = assignPointsToClusters(centroids, self.points)
    newCentroids = optimizeCentroids(centroids, self.points, clusters)
    
    initalIteration = True
    iterations = 0
    while not np.array_equal(centroids, newCentroids):
      if initalIteration:
        pass
        initalIteration = False
      else:
        centroids = newCentroids
        clusters = assignPointsToClusters(centroids, self.points)
        newCentroids = optimizeCentroids(centroids, self.points, clusters)
    
      iterations += 1
      yield centroids, clusters, iterations

  def _animate(self, generatorFuncData) -> None:
    '''
    Animates the kmeans algorithm
    i: current iteration
    points: n x d
    centroids: k x d
    clusters: assigned to clusters to points
    '''
    (centroids, clusters, i) = generatorFuncData 
    points = self.points
    plt.cla()
    plt.scatter(points[:, 0], points[:, 1], c=clusters, s=75, cmap='jet', marker='x')
    plt.scatter(centroids[:, 0], centroids[:, 1], edgecolors="black", c=assignPointsToClusters(centroids, centroids), cmap='jet', s=50)
    plt.title(f'Iteration: {i}\nK: {self.k}')

  def animate(self):
    fig, ax = plt.subplots()
    animObj = animation.FuncAnimation(
      fig,  
      self._animate, 
      self._generatorFuncAnimate,
      interval=500
    )

    animObj.save('kmeans.gif', writer='pillow', fps=5)
