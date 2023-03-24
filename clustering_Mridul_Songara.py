
# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the data file
fileData = pd.read_csv("dataset", header=None, delimiter= ' ')


# The first column of the datafile contains words which shall be clustered
# Since a word itself is not required and only its feature values are required for clustering, the first column of the datafile is dropped
dataset = fileData.drop(0, 'columns')

# File converted to numpy array for ease of matrix operations
dataset = dataset.to_numpy()





# The distace function
def distance(x, y):

    '''
    input: two vectors of equal length

    return: the Euclidean distance between the two vectors

    
    process followed:
    Take the norm 2 of the difference between the two input vectors.
    This gives Euclidean distance between the vectors
    '''
    
    return np.linalg.norm(x - y)





# The function for assignment of clusters to the objects of given dataset, based on the given cluster representatives
def assign(dataset, clusterRepresentatives):

    '''
    input: a dataset consisting of feature values of objects; 
        cluster representative - it refers to vectors assigned as cluster representatives

    return: a cluster value corresponding to each object - this value referes to the index of the cluster representive of the cluster in which the current object is assigned

    
    process followed:
    For each object calculate distance from each cluster representative.
    Assign this object the representative which is closest to it
    '''

    # The index numbers of 'clusters' array corresponds to the object at same index number in the 'dataset' matrix
    # This array shall contain information about the clusters assigned to each object in the dataset
    clusters = np.zeros((len(dataset)))


    for i, objectData in enumerate(dataset):

        # Calculate distance of this object from each cluster representative
        clusterDistance = [distance(objectData, c) for c in clusterRepresentatives]

        # Find the index number corresponding to the minimum distance
        # This index number also corresponds to the closest cluster representative
        # Thus, the array 'clusters' contain the index number of the cluster 
        # representative which is closest to this object
        clusters[i] = np.argmin(clusterDistance)

    
    return clusters





# The function for calculating cluster representatives, for given objects and clusters assigned to them
def optimise(dataset, clusters):

    '''
    input: a dataset consisting of feature values of objects;
        clusters - array consisting of an integer number for each object; all the objects which are assigned same number belong to one cluster

    return: a matrix of cluster representatives; each row corresponds to one cluster representative

    
    process followed:
    From the given 'clusters' array, extract the number of unique clusters.
    Then, initialise a matrix of cluster representatives with zeros.
    To find a new cluster representative, find mean of all the object
    which belong to that cluster.
    '''
    
    # Find the number of unique clusters by finding the length of array of the unique cluster numbers
    noOfClusters = len(np.unique(clusters))
    
    # Initialise the cluster representative - the number of rows refer to the number of cluster representative; the number of columns is same as the number of features of the dataset
    clusterRepresentatives = np.zeros((noOfClusters, dataset.shape[1]))

    # Each loop finds a new cluster representative
    for cIndex in range(noOfClusters):
        
        # Find the objects which are assigned to the current cluster
        # This matrix contain feature values of the objects in the
        # current cluster
        objectsInThisCluster = dataset[clusters == cIndex]

        # Find row-wise mean of all these objects which are assigned
        # to the current cluster
        # This gives, for each feature, mean over all the objects in this cluster
        clusterRepresentatives[cIndex] = np.mean(objectsInThisCluster, axis = 0)
    

    return clusterRepresentatives





# The function to find clusters iteratively; each iteration shall find a better clustering because each iteration involves an optimisation step
def clustering(dataset, clusterRepresentatives, MaxIter = 100):

    '''
    input: a dataset consisting of feature values of objects;
        cluster representatives - a matrix of initial cluster representatives; each row corresponds to one cluster representative
        MaxIter - maximum number of iterations to loop through the assignment and optimisation step of clustering

    return: the final clusters

    
    process followed:
    Based on initial cluster representative, assign a cluster to each object in the dataset.
    Then, iteratively: (for a fixed number of iterations)
        optimise the clustering by finding new cluster representatives
        based on these new cluster representatives, assign a new cluster to each object in the dataset
        check for a termination criteria - compare the new cluster assignment with the previous cluster assignment
            terminate the loop if there is no change in clustering
    '''
    
    # Assignment Phase
    # Assign cluster representative to each object (i.e., each row of features)
    clusters = assign(dataset, clusterRepresentatives)


    # Maximum number of iterations is used as a safety termination criteria - incase the code takes too long to reach an optimum

    # Another criteria of termination, i.e., when any object do not change clusters, is also implemented within this loop
    # In case, the loop takes too long to reach such an optimum, it should run till 'MaxIter' and then STOP
    for _ in range (MaxIter):

        # Optimisation Phase
        # Compute new representative as mean of the current clusters
        clusterRepresentatives = optimise(dataset, clusters)

        # Assignment Phase
        # Assign cluster representative to each object (i.e., each row of features)
        updatedClusters = assign(dataset, clusterRepresentatives)

        # Termination Criteria:
        # STOP when the older cluster and the updated cluster are equal, i.e., no object is assigned a new clusters
        if (clusters == updatedClusters).all():
            break

        else:
            # Store the updated clusters for checking the termination criteria in the next loop
            clusters = updatedClusters
        

    return clusters





# The function to cluster using k-means clustering
def k_means(dataset, k):

    
    '''
    input: a dataset consisting of feature values of objects;
        Number of clusters required

    return: the final clusters
    
    
    process followed:
    Randomly initialise the initial cluster representatives
    Call the clustering function with the dataset and these initial cluster representatives.
        Return of clustering function will be clusters of objects in the dataset
    '''

    # Initialisation Phase
    # initialRandomObjects = np.random.choice(range(dataset.shape[0]), size=k, replace=False)
    initialRandomObjects = np.random.choice(dataset.shape[0], size=k, replace=False)

    # Choose initial cluster representatives as randomly chosen objects from the dataset
    initialRepresentatives = dataset[initialRandomObjects]

    # Perform Clustering based on the inital representatives
    return clustering(dataset, initialRepresentatives)





# The function to cluster using k-means++ clustering
def k_means_plus_plus(dataset, k):

    '''
    input: a dataset consisting of feature values of objects;
        Number of clusters required

    return: the final clusters
    
    
    process followed:
    Randomly initialise one of the initial cluster representatives.
    Remaining initial cluster representative shall be selected interatively using the available initial cluster representative/s
        calculate probability of selection of each of the object in the dataset
            this probability is based on the distance of the object from its closest available initial cluster representative
            higher probability is assigned to the object which is farther from its closest cluster representative
            lower probability is assigned to the object which is near its closest cluster representative
            This ensures selection of well-spread out initial cluster representatives
        Using this probability choose a new initial cluster representative
    
    Call the clustering function with the dataset and these initial cluster representatives.
        Return of clustering function will be clusters of objects in the dataset
    '''

    # Initialisation Phase
    # initialRandomObject = np.random.choice(range(dataset.shape[0]), size=1, replace=False)
    initialRandomObject = np.random.choice(dataset.shape[0], size=1, replace=False)

    # Initialise the initial representative matrix with zeros
    initialRepresentatives = np.zeros((k, dataset.shape[1]))

    # Choose the first initial cluster representatives as a randomly chosen objects from the dataset
    initialRepresentatives[0] = dataset[initialRandomObject]

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
        currentClusters = assign(dataset, initialRepresentatives[:i])
        # distanceToRepresentative = np.zeros((len(features)))

        distanceToClosestRepresentative = np.array([distance(dataset[i], initialRepresentatives[int(currentClusters[i])]) for i in range(len(currentClusters))])

        rX_squared = np.square(distanceToClosestRepresentative)
        sum_rX_squared = np.sum(rX_squared)
        probabilityOfSelection = rX_squared / sum_rX_squared

        nextRandomObject = np.random.choice(range(dataset.shape[0]), size=1, p=probabilityOfSelection)
        initialRepresentatives[i] = dataset[nextRandomObject]
    c
    
    return clustering(dataset, initialRepresentatives)





# The function to select a leaf node for clustering
# This function is required for bisecting k-means, to assess the quality of current clustering and pick the worse of the clusters to split next
def selectedCluster(leafNodes, dataset):
    '''
    input: the leaf nodes, dataset
    return: the selected leaf node, most appropriate for split

    process followed:
    Calculate sum of squared distances between each object for in a leaf node (each leaf node correspond to one cluster)
    Repeat for all the leaf nodes
    Return the node number of the leaf corresponding to the maximum sum of squared distance
    '''

    squareSumDistance = np.zeros(len(leafNodes))

    for nodeNo, leaf in enumerate(leafNodes):
        for objectI in leaf:
            for objectJ in leaf:
                if objectJ > objectI:
                    squareSumDistance[nodeNo] += np.square(distance(dataset[objectI], dataset[objectJ]))
        
        
    chosenNode = squareSumDistance.argmax()
    
    return chosenNode





# The function to cluster using bisecting k-means clustering
def bisecting_k_means(dataset, s):
    '''
    This function is used to implement bisecting k-means algorithm on the given dataset;

    input:- dataset: the dataset which we wish to cluster, s: the number of leaf nodes (it sets the termination criteria for the hierarchical clustering)

    return:- the tree corresponding to the clusters at each level; This tree is in form of a list of lists; each list in the tree corresponds to 
            the clusters at the that level.
            The first level of the hierarchy, that is the single cluster of entire dataset is not a part of this tree, since it does not add any new information about clustering.
            The first element is a list of two clusters, which are the leaf nodes at the second level.
            The second element is a list of three clusters, which are the leaf nodes at the third level and so on.....
    
    The process followed is -
        The entire dataset is the first cluster and the root node of the tree.
        Use k-means method to find two clusters in this node. The clusters are represented as an array of integers, the indices with same integer value belongs in same cluster.
        Assign these two clusters as leaf nodes in the variable 'leafNodes'. 
        Then, loop through the following steps untill len(leafNodes) < s:
            Select the most appropriate leaf node to be clustered next. The selection criteria here uses sum of squared euclidean distances (SSD) between each object in the cluster.
                Calculate SSD for each leaf node cluster
                Select the cluster with largest SSD.
            The selected cluster is split using the k-means methods.
            This selected cluster is now removed from the leafNodes.
            The new clusters are added in 'leafNodes'.
    '''
    # use k-means to find two clusters of the dataset
    clusters = k_means(dataset, 2)

    # Initialise a list which shall store the clusters obtained at each level of heirarchy
    treeOfClusterings = []

    # Add in this list, the clustering obtained at the current level
    treeOfClusterings.append(clusters)

    # The two clusters obtained are leaf nodes
    # Segregate these two clusters
    # clusters == 0 OR clusters == 1 here just refer to the index of the corresponding cluster representative; the representatives are not used here but information about
    # common representative is used to split the objects into two clusters
    cluster0 = np.argwhere(clusters==0)
    cluster1 = np.argwhere(clusters==1)

    # Finally both these clusters are assigned as two new leaf nodes
    leafNodes = [[int(i) for i in cluster0], [int(j) for j in cluster1]]

    # Termination criteria; STOP when required number of clusters are obtained
    while len(leafNodes) < s:

        
        # Among all the leaf nodes, choose the most suitable for the next split
        # 'chosenNode' contain index in 'leafNode' list, of the cluster which shall be split next
        chosenNode = selectedCluster(leafNodes, dataset)
        

        # cluster the objects in this chosen node using k-means; k = 2
        newClusters = k_means(dataset[leafNodes[chosenNode]], 2)


        # These two new clusters are now leaf nodes
        # Segregate the two clusters
        # newClusters == 0 OR newClusters == 1 here just refer to the index of the corresponding cluster representative; the representatives are not used here but information about
        # common representative is used to split the objects into two clusters
        newClusters0 = np.argwhere(newClusters==0)
        newClusters1 = np.argwhere(newClusters==1)
        
        # Convert the 2-D array into 1-D array for next operation, where the index corresponding to the split leaf node is extracted
        newClusters0 = np.ravel(newClusters0)
        newClusters1 = np.ravel(newClusters1)
    

        # Extract the indices corresponding to the leaf node which is currently split;
        # the 'k-means' return clusters with cluster id's as '0' and '1'
        # But we require cluster id's of the cluster which is currently split
        # So that, the two new leaf nodes will contain the values which are there in the currently split leaf node
        newLeaf0 = [leafNodes[chosenNode][i] for i in newClusters0]
        newLeaf1 = [leafNodes[chosenNode][i] for i in newClusters1]

        # The chosen node is no longer a leaf node
        # The chosen node is now removed from leaf node list
        leafNodes.pop(chosenNode)


        # Finally both of the newly obtained clusters are assigned as two new leaf nodes
        leafNodes.append([int(i) for i in newLeaf0])
        leafNodes.append([int(i) for i in newLeaf1])

        # The following operation is used to obtain clustering as a single array, in which each index correspond to the object index in dataset and the corresponding 
        # array value is the cluster id in which this object belongs
        currentClusters = np.zeros(len(dataset))
        idx = 0
        for leaf in leafNodes:

            for i in leaf:
                # Assign values in cluster according to the index number of this leaf in the list of leaf nodes
                currentClusters[i] = idx
            
            idx+=1

        # Append the new clusters array
        # This corresponds to the clustering obtained at the current level of heirarchy
        treeOfClusterings.append(currentClusters)
        

    return treeOfClusterings



# This function is used to obtain a matrix, with distance between each object of the given dataset
def objectDistanceMatrix(dataset):

    '''
    input: datset
    return: a 2-D array with distance between each object of the dataset; the value at row 'i' and column 'j' corresponds to distance between object 'i' and 'j' of the given dataset
    
    process followed:

    use the euclidean distance to calculate distance between each object. The distance function created above in this program is used.
    '''

    noOfObjects = len(dataset)
    distMat = np.zeros((noOfObjects, noOfObjects))

    for i in range(noOfObjects):
        for j in range(i, noOfObjects):

            if i < j:
                distMat[i][j] = distance(dataset[i], dataset[j])
                distMat[j][i] = distMat[i][j]

    return distMat




# Calculate silhouette coefficient for given clustering on the given dataset and the distance matrix corresponding to the dataset
def silhouetteCoefficient(dataset, clusters, distMat):

    '''
    input: dataset, cluster: 1-D array with integer values corresponding to the cluster Ids in which each object of the dataset belong, 
        distmat: 2-D array containing distance between each object of the dataset

    return: mean silhouette coefficient


    process followed:
    For each object:
        loop through each of the unique cluster Id
            if the object belongs to this cluster:
                calculate 'a' as mean of distance between this object and all the other objects of the cluster
            else:
                calculate 'b' as mean of distance between this object and all the objects of this cluster
    '''

    noOfObjects = len(dataset)

    # Obtain the unique cluster id's and their corresponding cluster size
    clusterId, clusterSize = np.unique(clusters, return_counts=True)

    # Initialise the silihouette coefficient for each object
    silCoeff = np.zeros(noOfObjects)

    # Calculate the silihouette coefficient for each object
    for i in range(noOfObjects):
        
        # Initialise the 'b' and 'a' value for this object
        # 'b' is initialised infinity - it is compared with mean distances of the object from other clusters, the lower of the two values is then assigned as new 'b'
        b = np.inf
        a = 0

        # Loop through all the clusters; use unique cluster ids and the corresponding size (size is number of objects in this cluster)
        for cIndex, cSize in zip(clusterId, clusterSize):

            # If the object belongs to this cluster
            if clusters[i]  == cIndex:

                # 'a' can be calculated only if there are more than one objects in its cluster
                if cSize > 1:
                    a = np.sum(distMat[i][clusters == cIndex]) / (cSize - 1)

                # 'a' if there is only one object in this cluster, 'a' = 0
                else:
                    a = 0
            else:
                tempb = np.sum(distMat[i][clusters == cIndex]) / cSize
                if tempb < b:
                    b = tempb
        
        # If a = 0, silihouette coefficient = 0
        # It implies the object does not belong to a cluster
        if a == 0:
            silCoeff[i] = 0

        # else calculate the silihouette coefficient
        else:
            silCoeff[i] = (b - a) / np.max((b, a))
    
    return np.mean(silCoeff)
        






# The function to compare by plotting evaluation of a given clustering method over a given dataset and a given range of clusters to evaluate and compare
def clusteringEvaluation(clusterRange, dataset, clusteringMethod):
    
    '''
    input: the range of number of clusters; dataset; clustering method

    process followed:
        Obtain a distance matrix for given dataset.
        Check the clustering methd required.

        Obtain the clustering for required range of clusters
        Obtain average silihouette coefficient for each of these clusterings
        Plot the average silihouette coefficient against the number of clusters

    '''

    # Obtain distance matrix for the dataset
    distmat = objectDistanceMatrix(dataset)

    # Initilase the silihouette coefficent for each clustering as required by the given range of clusters
    datasetSilihouette = np.zeros(len(clusterRange))
    
    # For bisecting k-means clustering
    if clusteringMethod.__name__ == 'bisecting_k_means':
        
        # Obtain depth required for this heirarchical clustering
        depth = clusterRange[-1]

        # Obtain the clusterings till this depth
        # treeOfClusters is a list of lists containg clustering at each level in hierarchy
        treeOfClusters = clusteringMethod(dataset, depth)
        
        # Obtain and display the average silihouette coefficient for the clusterings at each level of hierarchy
        for i, clusters in enumerate(treeOfClusters):
            
            datasetSilihouette[i] = silhouetteCoefficient(dataset, clusters, distmat)
            
            print(f"s = {clusterRange[i]} clusters are formed with average silhouette coefficient : {np.round(datasetSilihouette[i], 2)}")

    else: # For other clustering methods
        
        # Loop to obtain clustering for each number of clusters 
        for i, k in enumerate(clusterRange):

            clusters = clusteringMethod(dataset, k)
            
            # Obtain and display the average silihouette coefficient for each of the clustering obtained
            datasetSilihouette[i] = silhouetteCoefficient(dataset, clusters, distmat)
            
            print(f"k = {k} clusters are formed with average silhouette coefficient : {np.round(datasetSilihouette[i], 2)}")
            i += 1

    # Plot the average silihouette coefficient against the number of clusters
    fig, ax = plt.subplots()
    ax.plot(clusterRange, datasetSilihouette)
    ax.set_title(f'{clusteringMethod.__name__} performance evaluation for different clusters')
    ax.set_xlabel('number of clusters (k)')
    ax.set_ylabel('Silhouette coefficient')




# Defined the range of number of clusters required for evaluation
clusterRange = range(2,10)

# Fix the seed to ensure same output every run
np.random.seed(20)


# Cluster using K-means method
print("Applying k-means clustering: ")
clusteringEvaluation(clusterRange, dataset, k_means)


# Cluster using K-means++ method
print("Applying k-means++ clustering: ")
clusteringEvaluation(clusterRange, dataset, k_means_plus_plus)


# Cluster using Bisecting K-means method
print("Applying bisecting k-means clustering: ")
clusteringEvaluation(clusterRange, dataset, bisecting_k_means)

# Show all the plots
plt.show()
