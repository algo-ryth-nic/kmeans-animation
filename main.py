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


class KmeansAnimate2D():
  def __init__(self, k: int, data: np.ndarray, startCentroids=None, columns:list=None, lables:list=None):
    """
    k: no. of clusters
    data: n x 2, n: number of points
    startCentroids (optional) = k x 2, k: number of clusters 
    columns (optional) = list of strings, x and y axis labels (length 2)
    lables (optional) = list of strings, labels for each cluster, order should match centroids
    """
    self.k = k
    
    # 2d data allowed for now 
    if data.shape[1] != 2:
      raise RuntimeError('data must be of shape n x 2')
    
    if not isinstance(data, np.ndarray):
      raise RuntimeError('data must be of type numpy.ndarray')

    self.points = data
    self.initialCentroids = startCentroids
    self.columns = columns
    self.cluster_lables = lables

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

  def _animate(self, generatorFuncData,) -> None:
    '''
    Animates the kmeans algorithm
    i: current iteration
    points: n x d
    centroids: k x d
    clusters: assigned clusters for points
    '''
    (centroids, clusters, i) = generatorFuncData 
    points = self.points
    
    plt.cla()
    cluster_scatter = plt.scatter(points[:, 0], points[:, 1], c=clusters, s=75, cmap='jet', marker='x')
    plt.scatter(centroids[:, 0], centroids[:, 1], edgecolors="black", c=assignPointsToClusters(centroids, centroids), cmap='jet', s=50)
    
    if not self.columns == None and len(self.columns) == 2 :
      plt.xlabel(self.columns[0])
      plt.ylabel(self.columns[1])
    
    if not self.cluster_lables == None and len(self.cluster_lables) == self.k :
      plt.legend(handles=cluster_scatter.legend_elements()[0], labels=self.cluster_lables)

    plt.title(f'Iteration: {i}\nK: {self.k}')

  def animate(self, interval=500) -> None:
    fig, ax = plt.subplots()
    animObj = animation.FuncAnimation(
      fig,  
      self._animate, 
      self._generatorFuncAnimate,
      interval=interval
    )
    
    self._animObj = animObj
    plt.show()


  def saveGIF(self, filename: str, fps=10) -> None:
    '''
    Saves the animation as a gif

    filename = name of the gif file, don't mention the extension
    '''
    self._animObj.save(filename, writer='pillow', fps=fps)
  
  def saveMP4(self, filename: str, fps=10) -> None:
    '''
    Saves the animation as a mp4

    filename = name of the mp4 file, don't mention the extension
    '''
    self._animObj.save(filename+'.mp4', writer='ffmpeg', fps=fps)


if __name__ == '__main__':
  # generate random clusters of data with 3000 samples using numpy
  points = np.random.rand(50, 2)
  kmeans = KmeansAnimate2D(k=3, data=points)
  kmeans.animate()