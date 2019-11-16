
import numpy

NOT_VISITED_LABEL = -99
NOISE_LABEL = -1

def dbscan(data, min_pts, epsilon):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
    
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [NOT_VISITED_LABEL]*len(data)

    # C is the ID of the current cluster.    
    cluster = 0
    
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for datum_idx in range(0, len(data)):
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[datum_idx] == NOT_VISITED_LABEL):
           continue
        
        # Find all of P's neighboring points.
        neighbors = neighbor_points(data, datum_idx, epsilon)
        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(neighbors) < min_pts:
            labels[datum_idx] = NOISE_LABEL
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           flood_fill(data, labels, datum_idx, neighbors, cluster, epsilon, min_pts)
           cluster += 1
    
    # All data has been clustered!
    return labels


def flood_fill(data, labels, datum_idx, neighbors, cluster, epsilon, min_pts):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[datum_idx] = cluster
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    for i in range(len(neighbors)):    
        # Get the next point from the queue.        
        neighbor = neighbors[i]
        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[neighbor] == NOISE_LABEL:
           labels[neighbor] = cluster
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[neighbor] == NOT_VISITED_LABEL:
            # Add Pn to cluster C (Assign cluster label C).
            labels[neighbor] = cluster
            
            # Find all the neighbors of Pn
            new_neighbors = neighbor_points(data, neighbor, epsilon)
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if len(new_neighbors) >= min_pts:
                neighbors = neighbors + new_neighbors
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts               
        
        # Advance to the next point in the FIFO queue.        
        i = i + 1
    
    # We've finished growing cluster C!


def neighbor_points(data, datum_idx, epsilon):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []
    
    # For each point in the dataset...
    for curr_idx in range(0, len(data)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if find_euclidean_distance(data[datum_idx], data[curr_idx]) < epsilon:
           neighbors.append(curr_idx)
        # print(find_euclidean_distance(data[datum_idx], data[curr_idx]))
            
    return neighbors

def find_euclidean_distance(a, b):
    dist_square = 0
    for i in range(len(a)):
        dist_square = dist_square + ((a[i] - b[i])**2)
    
    return dist_square**(.5)

data = [
    [5.1,3.5,1.4,0.2],
    [4.9,3.0,1.4,0.2],
    [4.7,3.2,1.3,0.2],
    [4.6,3.1,1.5,0.2],
    [5.0,3.6,1.4,0.2]
]

clusters = dbscan(data, epsilon=1, min_pts=3)
print(clusters)