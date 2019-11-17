class DBSCAN:
    def __init__(self, min_pts, epsilon):
        self.min_pts = min_pts
        self.epsilon = epsilon
        self.NOT_VISITED_LABEL = -99
        self.NOISE_LABEL = -1

    def label_dbscan(self, data):
        labels = [self.NOT_VISITED_LABEL]*len(data) 
        cluster = 0

        for datum_idx in range(0, len(data)):
            if not (labels[datum_idx] == self.NOT_VISITED_LABEL):
                continue

            neighbors = self.neighbor_points(data, datum_idx)

            if len(neighbors) < self.min_pts:
                labels[datum_idx] = self.NOISE_LABEL
            else: 
                self.flood_fill(data, labels, datum_idx, neighbors, cluster)
                cluster += 1

        return labels


    def flood_fill(self, data, labels, datum_idx, neighbors, cluster):
        labels[datum_idx] = cluster
        i = 0
        while i < len(neighbors):           
            neighbor = neighbors[i]
            
            if labels[neighbor] == self.NOISE_LABEL:
                labels[neighbor] = cluster
            elif labels[neighbor] == self.NOT_VISITED_LABEL:
                labels[neighbor] = cluster
                new_neighbors = self.neighbor_points(data, neighbor)
                if len(new_neighbors) >= self.min_pts:
                    neighbors = neighbors + new_neighbors
                    
            i = i + 1


    def neighbor_points(self, data, datum_idx):
        neighbors = []

        for curr_idx in range(0, len(data)):
            if self.find_euclidean_distance(data[datum_idx], data[curr_idx]) < self.epsilon:
                neighbors.append(curr_idx)
            # print(find_euclidean_distance(data[datum_idx], data[curr_idx]))

        return neighbors

    def find_euclidean_distance(self, a, b):
        dist_square = 0
        for i in range(len(a)):
            dist_square = dist_square + ((a[i] - b[i])**2)

        return dist_square**(.5)


import pandas as pd
import numpy as np


def pre_process(path_file):
    df = pd.read_csv(path_file, sep=',', names=['A1', 'A2', 'A3', 'A4', 'kelas'])
    
    obj_df = df.select_dtypes(include=['object']).copy()
    obj_df['kelas'] = obj_df['kelas'].astype('category')
    obj_df['kelas'] = obj_df['kelas'].cat.codes
    
    df['kelas'] = obj_df['kelas']
    
    return df

df = pre_process('iris.data')


X = df.drop(['kelas'], axis = 1)
y = df['kelas']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train.shape




dbscan = DBSCAN(min_pts = 5, epsilon = .9)
labels = dbscan.label_dbscan(X_train)
print(len(labels))
print(labels)

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

x = X_train[:,0]
y = X_train[:,1]
z = X_train[:,2]
t = X_train[:,3]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z, y, t, s=10)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
