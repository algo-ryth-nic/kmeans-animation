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
  '''
  n = points.shape[0] # no. of points
  k = centroids.shape[0] # no. of clusters``
  clusters = np.zeros(shape=(n,)) # n x 1, where n is the number of points

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

def kmeans(k: int, points: np.ndarray) -> np.ndarray:
  '''
  kmeans algorithm
  k: no. of clusters
  points: n x d, where is n the number of points & d is no. of dimensions (d=2 for now)
  '''
  centroids = initializeCentroids(k, points)
  clusters = assignPointsToClusters(centroids, points)
  newCentroids = optimizeCentroids(centroids, points, clusters)
  while not np.array_equal(centroids, newCentroids):
    old_centroids = newCentroids
    clusters = assignPointsToClusters(old_centroids, points)
    newCentroids = optimizeCentroids(old_centroids, points, clusters)
  return centroids, clusters

def generatorFuncAnimate(k, points):
  '''
  Animates the kmeans algorithm
  i: current iteration
  points: n x d, where is n the number of points & d is no. of dimensions (d=2 for now)
  centroids: k x d, where is k the number of clusters & d is no. of dimensions (d=2 for now)
  clusters: assigned to clusters to points
  colors: matplotlib color map
  '''
  centroids = initializeCentroids(k, points)
  clusters = assignPointsToClusters(centroids, points)
  newCentroids = optimizeCentroids(centroids, points, clusters)
  
  initalIteration = True
  iterations = 0
  while not np.array_equal(centroids, newCentroids):
    if initalIteration:
      pass
      initalIteration = False
    else:
      centroids = newCentroids
      clusters = assignPointsToClusters(centroids, points)
      newCentroids = optimizeCentroids(centroids, points, clusters)
   
    iterations += 1
    yield centroids, clusters, points, iterations

def animate(generatorFuncData) -> None:
  '''
  Animates the kmeans algorithm
  i: current iteration
  points: n x d, where is n the number of points & d is no. of dimensions (d=2 for now)
  centroids: k x d, where is k the number of clusters & d is no. of dimensions (d=2 for now)
  clusters: assigned to clusters to points
  colors: matplotlib color map
  '''
  (centroids, clusters, points, i) = generatorFuncData 
  n = points.shape[0] # no. of points
  k = centroids.shape[0] # no. of clusters
  # colors = np.array()
  plt.cla()
  plt.scatter(points[:, 0], points[:, 1], c=clusters, s=100, cmap='jet', marker='x')
  plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=25)
  # return plt
  plt.title(f'Iteration: {i}')


points = np.array([[ 15,  39],        [ 15,  81],        [ 16,   6],        [ 16,  77],        [ 17,  40],        [ 17,  76],        [ 18,   6],        [ 18,  94],        [ 19,   3],        [ 19,  72],        [ 19,  14],        [ 19,  99],        [ 20,  15],        [ 20,  77],        [ 20,  13],        [ 20,  79],        [ 21,  35],        [ 21,  66],        [ 23,  29],        [ 23,  98],        [ 24,  35],        [ 24,  73],        [ 25,   5],        [ 25,  73],        [ 28,  14],        [ 28,  82],        [ 28,  32],        [ 28,  61],        [ 29,  31],        [ 29,  87],        [ 30,   4],        [ 30,  73],        [ 33,   4],        [ 33,  92],        [ 33,  14],        [ 33,  81],        [ 34,  17],        [ 34,  73],        [ 37,  26],        [ 37,  75],        [ 38,  35],        [ 38,  92],        [ 39,  36],        [ 39,  61],        [ 39,  28],        [ 39,  65],        [ 40,  55],        [ 40,  47],        [ 40,  42],        [ 40,  42],        [ 42,  52],        [ 42,  60],        [ 43,  54],        [ 43,  60],        [ 43,  45],        [ 43,  41],        [ 44,  50],        [ 44,  46],        [ 46,  51],        [ 46,  46],        [ 46,  56],        [ 46,  55],        [ 47,  52],        [ 47,  59],        [ 48,  51],        [ 48,  59],        [ 48,  50],        [ 48,  48],        [ 48,  59],        [ 48,  47],        [ 49,  55],        [ 49,  42],        [ 50,  49],        [ 50,  56],        [ 54,  47],        [ 54,  54],        [ 54,  53],        [ 54,  48],        [ 54,  52],        [ 54,  42],        [ 54,  51],        [ 54,  55],        [ 54,  41],        [ 54,  44],        [ 54,  57],        [ 54,  46],        [ 57,  58],        [ 57,  55],        [ 58,  60],        [ 58,  46],        [ 59,  55],        [ 59,  41],        [ 60,  49],        [ 60,  40],        [ 60,  42],        [ 60,  52],        [ 60,  47],        [ 60,  50],        [ 61,  42],        [ 61,  49],        [ 62,  41],        [ 62,  48],        [ 62,  59],        [ 62,  55],        [ 62,  56],        [ 62,  42],        [ 63,  50],        [ 63,  46],        [ 63,  43],        [ 63,  48],        [ 63,  52],        [ 63,  54],        [ 64,  42],        [ 64,  46],        [ 65,  48],        [ 65,  50],        [ 65,  43],        [ 65,  59],        [ 67,  43],        [ 67,  57],        [ 67,  56],        [ 67,  40],        [ 69,  58],        [ 69,  91],        [ 70,  29],        [ 70,  77],        [ 71,  35],        [ 71,  95],        [ 71,  11],        [ 71,  75],        [ 71,   9],        [ 71,  75],        [ 72,  34],        [ 72,  71],        [ 73,   5],        [ 73,  88],        [ 73,   7],        [ 73,  73],        [ 74,  10],        [ 74,  72],        [ 75,   5],        [ 75,  93],        [ 76,  40],        [ 76,  87],        [ 77,  12],        [ 77,  97],        [ 77,  36],        [ 77,  74],        [ 78,  22],        [ 78,  90],        [ 78,  17],        [ 78,  88],        [ 78,  20],        [ 78,  76],        [ 78,  16],        [ 78,  89],        [ 78,   1],        [ 78,  78],        [ 78,   1],        [ 78,  73],        [ 79,  35],        [ 79,  83],        [ 81,   5],        [ 81,  93],        [ 85,  26],        [ 85,  75],        [ 86,  20],        [ 86,  95],        [ 87,  27],        [ 87,  63],        [ 87,  13],        [ 87,  75],        [ 87,  10],        [ 87,  92],        [ 88,  13],        [ 88,  86],        [ 88,  15],        [ 88,  69],        [ 93,  14],        [ 93,  90],        [ 97,  32],        [ 97,  86],        [ 98,  15],        [ 98,  88],        [ 99,  39],        [ 99,  97],        [101,  24],        [101,  68],        [103,  17],        [103,  85],        [103,  23],        [103,  69],        [113,   8],        [113,  91],        [120,  16],        [120,  79],        [126,  28],        [126,  74],        [137,  18],        [137,  83]]) 
fig, ax = plt.subplots()
animObj = animation.FuncAnimation(
    fig, 
    animate, 
    generatorFuncAnimate(3, points),
    repeat=True, 
    interval=500
    )
animObj.save('kmeans.gif', writer='imagemagick', fps=2)