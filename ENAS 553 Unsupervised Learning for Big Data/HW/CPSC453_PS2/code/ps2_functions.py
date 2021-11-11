# ps2_functions.py
# Jay S. Stanley III, Yale University, Fall 2018
# CPSC 453 -- Problem Set 2
#
# This script contains functions for implementing graph clustering and signal processing.
#

import numpy as np
import codecs
import json
from numpy import random
from numpy.core.fromnumeric import partition, size
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh
from sklearn import cluster
from sklearn.cluster import KMeans

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        my_array    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


def gaussian_kernel(X, kernel_type="gaussian", sigma=3.0, k=5):
    """gaussian_kernel: Build an adjacency matrix for data using a Gaussian kernel
    Args:
        X (N x d np.ndarray): Input data
        kernel_type: "gaussian" or "adaptive". Controls bandwidth
        sigma (float): Scalar kernel bandwidth
        k (integer): nearest neighbor kernel bandwidth
    Returns:
        W (N x N np.ndarray): Weight/adjacency matrix induced from X
    """
    _g = "gaussian"
    _a = "adaptive"

    kernel_type = kernel_type.lower()
    D = squareform(pdist(X))
    if kernel_type == "gaussian":  # gaussian bandwidth checking
        print("fixed bandwidth specified")

        if not all([type(sigma) is float, sigma > 0]):  # [float, positive]
            print("invalid gaussian bandwidth, using sigma = max(min(D)) as bandwidth")
            D_find = D + np.eye(np.size(D, 1)) * 1e15
            sigma = np.max(np.min(D_find, 1))
            del D_find
        sigma = np.ones(np.size(D, 1)) * sigma
    elif kernel_type == "adaptive":  # adaptive bandwidth
        print("adaptive bandwidth specified")

        # [integer, positive, less than the total samples]
        if not all([type(k) is int, k > 0, k < np.size(D, 1)]):
            print("invalid adaptive bandwidth, using k=5 as bandwidth")
            k = 5

        knnDST = np.sort(D, axis=1)  # sorted neighbor distances
        sigma = knnDST[:, k]  # k-nn neighbor. 0 is self.
        del knnDST
    else:
        raise ValueError

    W = ((D**2) / sigma[:, np.newaxis]**2).T
    W = np.exp(-1 * (W))
    W = (W + W.T) / 2  # symmetrize
    W = W - np.eye(W.shape[0])  # remove the diagonal
    return W


# BEGIN PS2 FUNCTIONS


def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """
    # generate the partion for N data
    partition_index = np.linspace(0, N, k+1)
    partition_index = np.ceil(partition_index).astype(int)
    partition_index_start = partition_index[:-1]
    partition_index_end = partition_index[1:]
    if not partition_index_start.shape[0] == k and partition_index_end.shape[0] == k:
        raise ValueError("Partition has incorrect shape")

    # generate ground truth
    gt = np.zeros((N, 1))
    for i in range(k):
        gt[partition_index_start[i]:partition_index_end[i]] = i
    
    # generate coords
    uniform_circle_sample_angle = np.linspace(0, 2*np.pi*k/(k+1), k)
    cluster_mean_x = np.sin(uniform_circle_sample_angle)
    cluster_mean_y = np.cos(uniform_circle_sample_angle)
    
    coords = np.random.normal(scale=sigma, size=(N,2))

    for i in range(k):
        coords[partition_index_start[i]:partition_index_end[i],0] += cluster_mean_x[i]
        coords[partition_index_start[i]:partition_index_end[i],1] += cluster_mean_y[i]

    # generate Adjacency    
    A = np.zeros((N,N))
    initial_prob = np.random.uniform(0, 1, size=(N,N))
    A = (initial_prob + initial_prob.T)/2

    for A_x_index in range(N):
        for A_y_index in range(N):
            flag = 0
            for i in range(k):
                if A_x_index in range(partition_index_start[i], partition_index_end[i]) and A_y_index in range(partition_index_start[i], partition_index_end[i]):
                    # if in the same cluster
                    flag = 1
            if flag == 1:
                A[A_x_index, A_y_index] = A[A_x_index, A_y_index] < pii
            else:
                A[A_x_index, A_y_index] = A[A_x_index, A_y_index] < pij

    A = A.astype(int)

    return A, gt, coords



def L(A, normalized=True):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    row_sum = np.sum(A,axis=1) # axis=1 is row sum
    D = np.diag(row_sum) # D is not distance matrix but the row sum diagonal matrix
    D_half_inv = np.sqrt(np.linalg.inv(D))

    if normalized:
        L = D_half_inv.dot(D-A).dot(D_half_inv)
    else:
        L = D - A
    return L


def compute_fourier_basis(L):
    """compute_fourier_basis: Laplacian Diagonalization

    Args:
        L (N x N np.ndarray): graph Laplacian

    Returns:
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    """
    e, psi = np.linalg.eigh(L)
    return e, psi


def gft(s, psi):
    """gft: Graph Fourier Transform (GFT)

    Args:
        s (N x d np.ndarray): Matrix of graph signals.  Each column is a signal.
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    Returns:
        s_hat (N x d np.ndarray): GFT of the data
    """
    s_hat = psi.T.dot(s)
    return s_hat


def filterbank_matrix(psi, e, h):
    """filterbank_matrix: build a filter matrix using the input filter h

    Args:
        psi (N x N np.ndarray): graph Laplacian eigenvectors
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        h (function handle): A function that takes in eigenvalues
        and returns values in the interval (0,1)

    Returns:
        H (N x N np.ndarray): Filter matrix that can be used in the form
        filtered_s = H@s
    """
    assert h in ['low-pass','high-pass','gaussian']
    c = 1/2 # threshold for low-pass and high-pass filter
    mu = -4
    sigma = 1

    e_filtered = np.zeros(e.shape)
    N = e.shape[0]
    if h == 'low-pass':
        for i in range(N):
            if e[i] < c:
                e_filtered[i] = 1
            else:
                e_filtered[i] = 0
    
    if h == 'high-pass':
        for i in range(N):
            if e[i] > c:
                e_filtered[i] = 1
            else:
                e_filtered[i] = 0

    if h =='gaussian':
        for i in range(N):
            e_filtered[i] = np.exp(-(e[i]-mu)**2/(2*sigma**2))

    H = psi.dot(np.diag(e_filtered)).dot(psi.T)

    return H


def kmeans(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    # init = kmeans_plusplus(X, k)  # find your initial centroids

    tol = 1e-5
    numOfSample = X.shape[0]
    n_dim = X.shape[1]

    cluster_distance = np.inf

    # repetition
    curr_best_centroid = np.zeros((k, n_dim))
    curr_best_labels = np.zeros((numOfSample, 1))

    for rep_idx in range(nrep):
        
        init_centroid = kmeans_plusplus(X, k) # kxd matrix
        curr_centroid = init_centroid
        for itr_idx in range(itermax):
            sample_label = np.zeros((numOfSample, 1))
            # get the label for each sample based on nearest distance with centroid
            for i in range(numOfSample):
                sample_dist = cdist(curr_centroid, X[i,:].reshape(1,n_dim)) # kx1 shape
                cluster_idx = np.where(sample_dist == np.min(sample_dist))
                if cluster_idx[0].shape[0] > 1:
                    # if multiple indices are returned, only select the first one (randomly select one as the distances are the same)
                    sample_label[i] = cluster_idx[0][0]
                else:
                    sample_label[i] = cluster_idx[0]

            # compute new centroid based on sample label
            new_centroid = np.zeros((k, n_dim))
            for cluster_idx in range(k):
                cnt = 0
                centroid_sum = np.zeros((1, n_dim))
                for sample_idx in range(numOfSample):
                    if sample_label[sample_idx] == cluster_idx:
                        cnt += 1
                        centroid_sum[0,:] += X[sample_idx,:]

                if not cnt == 0:
                    centroid_sum[0,:] /= cnt
                else:
                    centroid_sum[0,:] = curr_centroid[cluster_idx,:]

                new_centroid[cluster_idx,:] = centroid_sum[0,:]
            
            # stop updating if the difference is small
            if np.sum(np.sqrt(np.linalg.norm(curr_centroid - new_centroid))) < tol:
                break
            else:
                curr_centroid = new_centroid


        # calculate cluster distance
        curr_cluster_distance = 0
        for cluster_idx in range(k):
            for sample_idx in range(numOfSample):
                if sample_label[sample_idx] == cluster_idx:
                    curr_cluster_distance += np.sqrt(np.linalg.norm(curr_centroid[cluster_idx,:] - X[sample_idx,:]))

        if curr_cluster_distance < cluster_distance:
            curr_best_centroid = curr_centroid
            curr_best_labels = sample_label
            cluster_distance = curr_cluster_distance

    labels = curr_best_labels

    return labels


def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    # choose a random initial point
    numOfPoints = X.shape[0]
    init_idx = np.random.randint(0, numOfPoints)

    # compute the distance matrix
    D_matrix = squareform(pdist(X))

    # compute the relative distance to the randomly selected initial centroid
    init_dist = D_matrix[init_idx]

    # compute the pmf of the entire points relative to the initial centroid
    pmf = init_dist / sum(init_dist)

    # sample all k centroids based on the pmf
    all_centroids_idx = np.random.choice(np.arange(numOfPoints), size=k, replace=False, p=pmf)
    centroids = X[all_centroids_idx,:]

    return centroids


def SC(L, k, psi=None, nrep=5, itermax=300, sklearn=False):
    """SC: Perform spectral clustering 
            via the Ng method
    Args:
        L (np.ndarray): Normalized graph Laplacian
        k (integer): number of clusters to compute
        nrep (int): Number of repetitions to average for final clustering
        itermax (int): Number of iterations to perform before terminating
        sklearn (boolean): Flag to use sklearn kmeans to test your algorithm
    Returns:
        labels (N x 1 np.array): Learned cluster labels
    """
    if psi is None:
        # compute the first k elements of the Fourier basis
        # use scipy.linalg.eigh (this will give flip the sign between psi norm matrix)
        e, psi = np.linalg.eigh(L)
        psi_k = psi[:, :k]
        pass
    else:  # just grab the first k eigenvectors
        psi_k = psi[:, :k]

    # normalize your eigenvector rows
    psi_k_rownorm = np.linalg.norm(psi_k, axis=1, ord=2)

    psi_norm = psi_k / psi_k_rownorm.reshape((psi_k_rownorm.shape[0],1))

    if sklearn:
        labels = KMeans(n_clusters=k, n_init=nrep,
                        max_iter=itermax).fit_predict(psi_norm)
    else:
        labels = kmeans(psi_norm, k, nrep=nrep, itermax=itermax)
        pass
        # your algorithm here

    return labels

