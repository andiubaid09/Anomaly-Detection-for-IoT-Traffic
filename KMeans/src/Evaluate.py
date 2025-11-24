import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib
from google.colab import files

