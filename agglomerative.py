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
        ii = 0
        jj = 1
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if i != j and matrix[i][j] < min_dist:
                    min_dist = matrix[i][j]
                    ii = i
                    jj = j
                    matrix[i][j] = INF
                    matrix[j][i] = INF

        return matrix, ii, jj, min_dist

    def fit(self, data):
        print(data)
        clusters = [Node(_) for _ in range(len(data))]
        matrix = self.get_distance_matrix(data)

        cluster = len(data)
        while cluster > self.n_cluster:
            matrix, i, j, dist = self.get_closest_point(matrix)
            print("closest point i " + str(i) + ", j " + str(j))
            endcluster = clusters[i]
            while endcluster.next != None:
                endcluster = endcluster.next
            endcluster1 = clusters[j]
            while endcluster1.next != None:
                endcluster1 = endcluster1.next
            if endcluster != endcluster1:
                endcluster.next = endcluster1
                cluster = cluster - 1
            print("cluster "+str(cluster))


        for _ in clusters:
            print(_.tostr())

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
        print(classes)


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
            

def main():
    data = [[1,2],
            [5,5],
            [6,6],
            [7,1],
            [2,5],
            [100,100],
            [100,100],
            [100000,10000]]

    model = Agglomerative(2)
    model.fit(data)
    classes = model.predict(data)

    print(classes)


if __name__ == "__main__":
    main()
