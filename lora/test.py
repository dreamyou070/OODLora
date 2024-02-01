import torch
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

mahalanobis_dists = [3,6,1,3,3]
mahalanobis_dists.sort()
figure = plt.figure()
plt.hist(mahalanobis_dists, bins=3)
plt.savefig('histogram.png')

mahalanobis_dists = [6,6,1,3,3]
mahalanobis_dists.sort()
figure = plt.figure()
plt.hist(mahalanobis_dists, bins=3)
plt.savefig('histogram2.png')
