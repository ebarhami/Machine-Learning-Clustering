# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from math import *




class Node:
    def __init__(self, name):
        self.name = str(name)
        self.next = None

    def tostr(self):
        nextnode = 'NULL' if not self.next else self.next.name
        return str(self.name) + '->' + nextnode


class Agglomerative:
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster

    def get_distance_matrix(self, data):
        matrix = []
        for i in range(len(data)):
            tmp = []
            for j in range(len(data)):
                tmp.append(self.get_dist(data[i], data[j]))
            matrix.append(tmp)

        return matrix
    
    def get_dist(self, p1, p2):
        dist = 0
        for i in range(len(p1)):
            dist = dist + (p1[i]-p2[i])**2

        return sqrt(dist)

    def get_closest_point(self, matrix):
        INF = 99999999
        min_dist = INF
        ii = -1
        jj = -1
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                if i != j and matrix[i][j] < min_dist:
                    min_dist = matrix[i][j]
                    ii = i
                    jj = j
        matrix[ii][jj] = INF
        matrix[jj][ii] = INF
        return matrix, ii, jj, min_dist

    def fit(self, data):
        clusters = [Node(_) for _ in range(len(data))]
        matrix = self.get_distance_matrix(data)

        cluster = len(data)
        a = 0
        while cluster > self.n_cluster:
            a = a+1
            matrix, i, j, dist = self.get_closest_point(matrix)
            endcluster = clusters[i]
            while endcluster.next != None:
                endcluster = endcluster.next
            endcluster1 = clusters[j]
            while endcluster1.next != None:
                endcluster1 = endcluster1.next
            if endcluster != endcluster1:
                endcluster.next = endcluster1
                cluster = cluster - 1

        classes = []
        for i in range(len(data)):
            clas = clusters[i]
            while clas.next != None:
                clas = clas.next
            classes.append(clas.name)

        uniq = list(set(classes))
        mapped = {}
        for i in range(len(uniq)):
            mapped[uniq[i]] = i

        for i in range(len(classes)):
            classes[i] = mapped[classes[i]]
        
        self.data = data
        self.classes = classes


    def predict(self, data):
        classes = []
        for x in data:
            ans = -1
            cur_dist = 99999999
            for i in range(len(self.data)):
                if self.get_dist(x, self.data[i]) < cur_dist:
                    cur_dist = self.get_dist(x, self.data[i])
                    ans = self.classes[i]
            classes.append(ans)
        return classes


# %%
class K_Means:
    def __init__(self, clusters):
        self.clusters = clusters
        self.max_iter = 100
    
    def fit(self, data):        
        # Init centroids
        self.centroids = data.copy()
        np.random.shuffle(self.centroids)
        self.centroids = self.centroids[:self.clusters]
        
        for i in range(self.max_iter):
            distance = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
            classes = np.argmin(distance, axis=0)
            self.centroids = np.array([data[classes==k].mean(axis=0) for k in range(self.centroids.shape[0])])
            
    def predict(self, data):
        distance = np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))
        classes = np.argmin(distance, axis=0)
        return classes


# %%

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
    
    def predict(self, X_train, labels, X_test):
        pred_labels = []
        for point_test in X_test:
            dist = 999999
            idx = -1
            i = 0
            for point in X_train:
                temp = self.find_euclidean_distance(point_test, point)
                if temp < dist:
                    dist = temp
                    idx = i 
                i = i + 1
                
            pred_labels.append(labels[idx])
        
        return pred_labels
                


# %%
def pre_process(path_file):
    df = pd.read_csv(path_file, sep=',', names=['A1', 'A2', 'A3', 'A4', 'kelas'])
    
    obj_df = df.select_dtypes(include=['object']).copy()
    obj_df['kelas'] = obj_df['kelas'].astype('category')
    obj_df['kelas'] = obj_df['kelas'].cat.codes
    
    df['kelas'] = obj_df['kelas']
    
    return df


# %%
import pandas as pd
import numpy as np

df = pre_process('iris.data')
df


# %%
# Split into train and test data

X = df.drop(['kelas'], axis = 1)
y = df['kelas']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train.shape


# %%
agglomerative = Agglomerative(3)
agglomerative.fit(X_train)

classes = agglomerative.predict(X_test)
print(classes)


# %%
from sklearn.metrics import accuracy_score, f1_score, recall_score


pred = [0]*6

pred[0] = classes
pred[1] = [0 if x==0 else 2 if x==1 else 1 for x in pred[0]]
pred[2] = [1 if x==0 else 0 if x==1 else 2 for x in pred[0]]
pred[3] = [1 if x==0 else 2 if x==1 else 0 for x in pred[0]]
pred[4] = [2 if x==0 else 0 if x==1 else 1 for x in pred[0]]
pred[5] = [2 if x==0 else 1 if x==1 else 0 for x in pred[0]]

max_pred = [accuracy_score(y_test, pred[0]), accuracy_score(y_test, pred[1]), accuracy_score(y_test, pred[2]),
           accuracy_score(y_test, pred[3]), accuracy_score(y_test, pred[4]), accuracy_score(y_test, pred[5])]

# Find index with maximum accuracy
best_idx = max_pred.index(max(max_pred))

print(f"accuracy_score scratch:  {accuracy_score(y_test, pred[best_idx])}")
print(f"f1_score scratch: {f1_score(y_test, pred[best_idx], average=None)}")
print(f"recall_score scratch: {recall_score(y_test, pred[best_idx], average=None)}")


# %%
from sklearn.cluster import AgglomerativeClustering

def get_dist(p1, p2):
    dist = 0
    for i in range(len(p1)):
        dist = dist + (p1[i]-p2[i])**2
    return sqrt(dist)

clustering = AgglomerativeClustering().fit(X_train)
 
AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                        connectivity=None, distance_threshold=None,
                        linkage='ward', memory=None, n_clusters=3,
                        pooling_func='deprecated')
clustering.labels_

labels = []
for x in X_test:
    ans = -1
    cur_dist = 99999999
    for i in range(len(X_train)):
        if get_dist(x, X_train[i]) < cur_dist:
            cur_dist = get_dist(x, X_train[i])
            ans = clustering.labels_[i]
    labels.append(ans)

pred = [0]*6

pred[0] = labels
pred[1] = [0 if x==0 else 2 if x==1 else 1 for x in pred[0]]
pred[2] = [1 if x==0 else 0 if x==1 else 2 for x in pred[0]]
pred[3] = [1 if x==0 else 2 if x==1 else 0 for x in pred[0]]
pred[4] = [2 if x==0 else 0 if x==1 else 1 for x in pred[0]]
pred[5] = [2 if x==0 else 1 if x==1 else 0 for x in pred[0]]

max_pred = [accuracy_score(y_test, pred[0]), accuracy_score(y_test, pred[1]), accuracy_score(y_test, pred[2]),
           accuracy_score(y_test, pred[3]), accuracy_score(y_test, pred[4]), accuracy_score(y_test, pred[5])]

# Find index with maximum accuracy
best_idx = max_pred.index(max(max_pred))

print(f"accuracy_score library:  {accuracy_score(y_test, pred[best_idx])}")
print(f"f1_score library: {f1_score(y_test, pred[best_idx], average=None)}")
print(f"recall_score library: {recall_score(y_test, pred[best_idx], average=None)}")


# %%
kmeans_scratch = K_Means(3)
kmeans_scratch.fit(X_train)


# %%
from sklearn.metrics import accuracy_score, f1_score, recall_score

y_pred = kmeans_scratch.predict(X_test)

pred = [0]*6

pred[0] = y_pred
pred[1] = [0 if x==0 else 2 if x==1 else 1 for x in pred[0]]
pred[2] = [1 if x==0 else 0 if x==1 else 2 for x in pred[0]]
pred[3] = [1 if x==0 else 2 if x==1 else 0 for x in pred[0]]
pred[4] = [2 if x==0 else 0 if x==1 else 1 for x in pred[0]]
pred[5] = [2 if x==0 else 1 if x==1 else 0 for x in pred[0]]

max_pred = [accuracy_score(y_test, pred[0]), accuracy_score(y_test, pred[1]), accuracy_score(y_test, pred[2]),
           accuracy_score(y_test, pred[3]), accuracy_score(y_test, pred[4]), accuracy_score(y_test, pred[5])]

# Find index with maximum accuracy
best_idx = max_pred.index(max(max_pred))

print(f"accuracy_score scratch:  {accuracy_score(y_test, pred[best_idx])}")
print(f"f1_score scratch: {f1_score(y_test, pred[best_idx], average=None)}")
print(f"recall_score scratch: {recall_score(y_test, pred[best_idx], average=None)}")


# %%



# %%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(X_train)


# %%
kmeans_pred = kmeans.predict(X_test)

print(kmeans_pred)
print(y_test)


# %%
pred = [0]*6

pred[0] = kmeans_pred
pred[1] = [0 if x==0 else 2 if x==1 else 1 for x in pred[0]]
pred[2] = [1 if x==0 else 0 if x==1 else 2 for x in pred[0]]
pred[3] = [1 if x==0 else 2 if x==1 else 0 for x in pred[0]]
pred[4] = [2 if x==0 else 0 if x==1 else 1 for x in pred[0]]
pred[5] = [2 if x==0 else 1 if x==1 else 0 for x in pred[0]]

max_pred = [accuracy_score(y_test, pred[0]), accuracy_score(y_test, pred[1]), accuracy_score(y_test, pred[2]),
           accuracy_score(y_test, pred[3]), accuracy_score(y_test, pred[4]), accuracy_score(y_test, pred[5])]

# Find index with maximum accuracy
best_idx = max_pred.index(max(max_pred))

print(f"accuracy_score library:  {accuracy_score(y_test, pred[best_idx])}")
print(f"f1_score library: {f1_score(y_test, pred[best_idx], average=None)}")
print(f"recall_score library: {recall_score(y_test, pred[best_idx], average=None)}")


# %%
dbscan = DBSCAN(min_pts = 10, epsilon = .9)
labels = dbscan.label_dbscan(X_train)
print(labels)

pred = [0]*6

pred[0] = dbscan.predict(X_train, labels, X_test)
pred[1] = [0 if x==0 else 2 if x==1 else 1 for x in pred[0]]
pred[2] = [1 if x==0 else 0 if x==1 else 2 for x in pred[0]]
pred[3] = [1 if x==0 else 2 if x==1 else 0 for x in pred[0]]
pred[4] = [2 if x==0 else 0 if x==1 else 1 for x in pred[0]]
pred[5] = [2 if x==0 else 1 if x==1 else 0 for x in pred[0]]

max_pred = [accuracy_score(y_test, pred[0]), accuracy_score(y_test, pred[1]), accuracy_score(y_test, pred[2]),
           accuracy_score(y_test, pred[3]), accuracy_score(y_test, pred[4]), accuracy_score(y_test, pred[5])]

# Find index with maximum accuracy
best_idx = max_pred.index(max(max_pred))

print(f"accuracy_score scratch:  {accuracy_score(y_test, pred[best_idx])}")
print(f"f1_score scratch: {f1_score(y_test, pred[best_idx], average=None)}")
print(f"recall_score scratch: {recall_score(y_test, pred[best_idx], average=None)}")


# %%
from sklearn.cluster import DBSCAN as DBSCAN2

clustering = DBSCAN2(eps=.9, min_samples=10).fit(X_train)
print(clustering.labels_)

dbscan2 = DBSCAN(min_pts = 10, epsilon = .9)

pred2 = [0]*6
pred2[0] = dbscan2.predict(X_train, clustering.labels_, X_test)
pred2[1] = [0 if x==0 else 2 if x==1 else 1 if x==2 else -1 for x in pred2[0]]
pred2[2] = [1 if x==0 else 0 if x==1 else 2 if x==2 else -1 for x in pred2[0]]
pred2[3] = [1 if x==0 else 2 if x==1 else 0 if x==2 else -1 for x in pred2[0]]
pred2[4] = [2 if x==0 else 0 if x==1 else 1 if x==2 else -1 for x in pred2[0]]
pred2[5] = [2 if x==0 else 1 if x==1 else 0 if x==2 else -1 for x in pred2[0]]

max_pred2 = [accuracy_score(y_test, pred2[0]), accuracy_score(y_test, pred2[1]), accuracy_score(y_test, pred2[2]),
           accuracy_score(y_test, pred2[3]), accuracy_score(y_test, pred2[4]), accuracy_score(y_test, pred2[5])]

# Find index with maximum accuracy
best_idx2 = max_pred.index(max(max_pred2))

print(f"accuracy_score library:  {accuracy_score(y_test, pred[best_idx2])}")
print(f"f1_score library: {f1_score(y_test, pred[best_idx2], average=None)}")
print(f"recall_score library: {recall_score(y_test, pred[best_idx2], average=None)}")


