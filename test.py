import csv
import sys
from time import time

import pyspark as ps
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
import numpy as np

from dbscan import DBSCAN
from metrics import geo_distance


#  if __name__ == '__main__':
    #  #  i = int(sys.argv[1])
    #  #  centers = [[1, 1], [-1, -1], [1, -1]]
    #  #  samples = [750, 7500, 75000, 750000, 7500000]
    #  #  eps = [0.3, 0.1, 0.03, 0.01, 0.003]
    #  #  n_part = [16, 128, 1024, 8192, 65536]
    #  #  sc = ps.SparkContext()
    #  #  X, labels_true = make_blobs(n_samples=samples[i], centers=centers,
                                #  #  cluster_std=0.4,
                                #  #  random_state=0)

    #  #  X = StandardScaler().fit_transform(X)

    #  #  test_data = sc.parallelize(enumerate(X))
    #  #  start = time()
    #  #  dbscan = DBSCAN(eps[i], 10, max_partitions=n_part[i])
    #  #  dbscan.train(test_data)
    #  #  result = np.array(dbscan.assignments())
    #  #  run_time = time() - start
    #  #  with open('benchmark.csv', 'w') as f:
        #  #  f.write('\n%i,%f,%i,%i' % (samples[i], eps[i], n_part[i], run_time))

    #  i = int(sys.argv[1])
    #  centers = [[1, 1], [-1, -1], [1, -1]]
    #  samples = [750, 7500, 75000, 750000, 7500000]
    #  eps = [0.3, 0.1, 0.03, 0.01, 0.003]
    #  n_part = [16, 128, 1024, 8192, 65536]
    #  sc = ps.SparkContext()
    #  X, labels_true = make_blobs(n_samples=samples[i], centers=centers,
                                #  cluster_std=0.4,
                                #  random_state=0)
    #  X = StandardScaler().fit_transform(X)

    #  test_data = sc.parallelize(enumerate(X))

    #  #  distance_matrix = squareform(pdist(test_data, (lambda u,v: geo_distance(u,v))))
    #  dbscan = DBSCAN(eps=eps[i], min_samples=3, max_partitions=n_part[i], metric='precomputed')
    #  dbscan.train(test_data)

    #  # Plot
    #  #  fig = plt.figure(1)
    #  #  col = 'k'
    #  #  plt.plot(X[:, 0], X[:, 1], '*', markerfacecolor='k', markeredgecolor='k', markersize=5)

    #  #  core_samples_mask = np.zeros_like(labels, dtype=bool)
    #  #  core_samples_mask[db.core_sample_indices_] = True

    #  #  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    #  #  unique_labels = set(labels)
    #  #  colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    #  #  fig = plt.figure(2)
    #  #  center_point = []
    #  #  heatmap_point = []



if __name__ == '__main__':
    # Example of pypadis.DBSCAN
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    from time import time
    from itertools import izip
    import os

    X = []
    with open("staypoints.csv","rb") as f:
        reader = csv.reader(f)
        for line in reader:
            X.append(line)
        X = np.array(X, np.float)

    sc = ps.SparkContext()
    test_data = sc.parallelize(enumerate(X))
    start = time()
    dbscan = DBSCAN(eps=0.02, min_samples=20, metric='precomputed')
    dbscan.train(test_data)
    result = np.array(dbscan.assignments())
    print 'clusters count: %s' % len(set(result[: 1]))
    import pdb; pdb.set_trace()
