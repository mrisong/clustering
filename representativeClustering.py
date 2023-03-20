# Import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def distance(x, y):
    return np.linalg.norm(x - y)




def assign(features, clusterRepresentatives):

    '''
    For each feature calculate distance from each cluster representative.
    Assign this feature the representative which is closest to it
    '''
    # The index numbers of clusters array corresponds to the object
    # at same index number in the feature matrix
    # This array shall contain information about the cluster 
    # assigned to the corresponding object in the feature matrix
    clusters = np.zeros((len(features)))

    if clusterRepresentatives.ndim == 1:
        clusterRepresentatives = np.array([clusterRepresentatives])

    for fIndex in range(len(features)):

        # Calculate distance of this feature from each cluster representative
        clusterDistance = [distance(features[fIndex], c) for c in clusterRepresentatives]

        # Find the index number corresponding to the minimum distance
        # This index number also corresponds to the closest cluster representative
        # Thus, the array 'clusters' contain the index number of the cluster 
        # representative which is closest to this object
        clusters[fIndex] = np.argmin(clusterDistance)

    # print(clusters[:5])
    # print('abcdef')
        
    return clusters




def optimise(features, clusters, clusterRepresentatives):

    '''
    For each cluster, find mean of features of all the object
    which are assigned to it. This mean shall be designated
    as the new cluster representative for that cluster
    Return the array of the new cluster representatives
    '''

    for cIndex in range(len(clusterRepresentatives)):
        
        # Find the objects which are assigned to the current cluster
        # This matrix contain feature values of the objects in the
        # current cluster
        objectsInThisCluster = features[clusters == cIndex]

        # Find row-vise mean of all these objects which are assigned
        # to the current cluster
        # This gives, for each feature, mean over all the objects in this cluster
        clusterRepresentatives[cIndex] = np.mean(objectsInThisCluster, axis = 0)
    

    return clusterRepresentatives




def clustering(features, clusterRepresentatives):

    # Assignment Phase
    # Assign cluster representative to each object (i.e., each row of features)
    clusters = assign(features, clusterRepresentatives)
    
    
    # Optimisation Phase
    # Compute new representative as mean of the current clusters
    clusterRepresentatives = optimise(features, clusters, clusterRepresentatives)

    return clusters, clusterRepresentatives




def k_means(features, k):

    # Initialisation Phase
    np.random.seed(42)
    initialRandomObjects = np.random.choice(range(features.shape[0]), size=k)

    # Choose initial cluster representatives as randomly chosen objects from the dataset
    initialRepresentatives = features[initialRandomObjects]

    # Perform Clustering based on the inital representatives
    return clustering(features, initialRepresentatives)
    

    




def k_means_plus_plus(features, k):

    # Initialisation Phase
    np.random.seed(42)
    initialRandomObject = np.random.choice(range(features.shape[0]), size=1)

    # Initialise the initial representative matrix with zeros
    initialRepresentatives = np.zeros((k, features.shape[1]))

    # Choose the first initial cluster representatives as a randomly chosen objects from the dataset
    initialRepresentatives[0] = features[initialRandomObject]

    # Finding a new representative for each cluster
    for i in range(1, k):
        # Find the closest representative - This basically forms a cluster
        # Following are the steps to choose closest representative
            # Find distance of every object from each representative
            # Assign the object to closest representative
            # Store the distance from the closest representative only

        # The distance from the closest representative shall be used for next step
        # Choose next representative with probability proportional to
        # distance squared
        
        # Form a cluster using the already chosen representatives
        currentClusters = assign(features, initialRepresentatives[:i])
        # distanceToRepresentative = np.zeros((len(features)))

        distanceToClosestRepresentative = np.array([distance(features[i], initialRepresentatives[int(currentClusters[i])]) for i in range(len(currentClusters))])

        rX_squared = np.square(distanceToClosestRepresentative)
        sum_rX_squared = np.sum(rX_squared)
        probabilityOfSelection = rX_squared / sum_rX_squared

        nextRandomObject = np.random.choice(range(features.shape[0]), size=1, p=probabilityOfSelection)
        initialRepresentatives[i] = features[nextRandomObject]
    
    
    return clustering(features, initialRepresentatives)
