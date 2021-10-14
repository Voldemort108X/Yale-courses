# ps1_functions.py
# CPSC 553 -- Problem Set 1
#
# This script contains uncompleted functions for implementing diffusion maps.
#
# NOTE: please keep the variable names that I have put here, as it makes grading easier.

# import required libraries
import numpy as np
import codecs, json
##############################
# Predefined functions
##############################

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


##############################
# Skeleton code (fill these in)
##############################


def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

    '''
    n, p = X.shape[0], X.shape[1]
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(X[i,:]-X[j,:])
    # return distance matrix
    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''
    n = D.shape[0]
    W = np.zeros((n,n))
    assert kernel_type in ['gaussian', 'adaptive']
    if kernel_type == 'gaussian':
        W = np.exp(-D**2/sigma**2)

    if kernel_type == 'adaptive':
        assert k > 0 and type(k) == int
        for i in range(n):
            knn_i = np.sort(D[i,:])
            for j in range(n):
                knn_j = np.sort(D[j,:])
                W[i,j] = 1/2*(np.exp(-D[i,j]**2/knn_i[k]**2)+np.exp(-D[i,j]**2/knn_j[k]**2)) # instead of k-1 as self-distance is not neighbor

    # return the affinity matrix
    return W


def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:

        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix

        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''

    # return the info for diffusion maps
    row_sum = np.sum(W,axis=1) # axis=1 is row sum
    D = np.diag(row_sum) # D is not distance matrix but the row sum diagonal matrix
    D_half_inv = np.sqrt(np.linalg.inv(D))
    Ms = D_half_inv.dot(W).dot(D_half_inv)
    eigValue, eigVector = np.linalg.eigh(Ms)
    index_sort = np.argsort(-eigValue)
    # sort the eigenvalue to obtain the largest-k components
    eigValue_sorted = eigValue[index_sort]
    eigVector_sorted = eigVector[:, index_sort]

    diff_vec = D_half_inv.dot(eigVector_sorted[:,1:])/np.linalg.norm(D_half_inv.dot(eigVector_sorted[:,1:]),axis=0)
    diff_eig = eigValue_sorted[1:]

    return diff_vec, diff_eig


def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix

    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t
    '''
    n = diff_vec.shape[0]
    diff_map = np.zeros((n,n-1))
    for i in range(n-1):
        diff_map[:,i] = diff_eig[i]**t*diff_vec[:,i]

    return diff_map

