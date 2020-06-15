### Load libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist
import random
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

#### GLOBAL VARIABLES
ITERATION = 10
CLUSTER_THRESHOLD = 0.1

class runClusters:
    """
    A class for different clustering methods algorithm
    """
    def __init__(self, data, label):
        self.data = data
        self.label = label

    #############################
    ### K-Means clustering ###
    #############################

    def kmeans(self, n_cluster, iters=ITERATION, tol = 1e-6):
        """
        K-means clustering algorithm:
        1. Start with $n_cluster$ centers with labels $0, 1, \ldots, k-1$
        2. Find the distance of each data point to each center
        3. Assign the data points nearest to a center to its label
        4. Use the mean of the points assigned to a center as the new center
        5. Repeat for a fixed number of iterations or until the centers stop changing
        """
        r, c = self.data.shape
        ### Random select n_cluster points as the initial points
        ### Define a score to check goodness of fit, it can be the sum of all points to its center
        best_score = np.infty
        for i in range(iters):
            centers = self.data[np.random.choice(r, n_cluster, replace=False)]
            delta  = np.infty
            while delta > tol:
                ### cdist calculates distance betweem two vectors
                ### ith column is the distance between ith point  and centers
                dis = cdist(self.data, centers)
                mdis = np.argmin(dis, axis = 1)
                new_centers = np.array([np.mean(self.data[mdis == i], axis = 0) for i in range(n_cluster)])
                delta = np.sum((new_centers - centers)**2)
                centers = new_centers
            scores = dis[mdis].sum()
            if scores < best_score:
                best_score = scores
                best_label = mdis
                best_centers = centers
        return best_label, best_centers

    #############################
    ### K-means ++ clustering ###
    #############################

    def k_means_pp(self, n_clusters, tol = 1e-6):
        """
        Choose one center uniformly at random from among the data points.
        For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
        Repeat Steps 2 and 3 until k centers have been chosen.
        Now that the initial centers have been chosen, proceed using standard k-means clustering
        """
        X_cp = self.data.copy()
        r, c  = X_cp.shape
        #### initialize 1 center
        centers = []
        center = X_cp[np.random.choice(r,1)]
        centers.append(center)
        dist = cdist(X_cp, center)
        delta = np.infty
        while len(centers) < n_clusters:
            probability = (dist**2/np.sum(dist**2)).flatten()
            center = X_cp[np.random.choice(r,1,p = probability)]
            centers.append(center)
            dist1 = cdist(X_cp,center)
            dist = np.c_[dist, dist1].min(axis=1).reshape(-1,1)
        while delta > tol:
            dis = cdist(X_cp, np.array(centers).reshape(-1,2))
            mdis = np.argmin(dis, axis=1)
            new_centers = np.array([np.mean(self.data[mdis == i], axis=0) for i in range(n_clusters)])
            delta = np.sum((new_centers - centers) ** 2)
            centers = new_centers
            labels = mdis
        return labels, centers

    ###################################
    ### Gaussian Mixture clustering ###
    ###################################

    def gaussian_mix(self, clusters, tol = 1e-3):
        """
        Extension of K-means
        1. Cluster modeled as gaussian distribution
        2. EM algorithm: Assign data to cluster with some probability
        In general, GMMs try to learn each cluster as a different Gaussian distribution. It assumes the data is generated from a limited mixture of Gaussians.
        In the presence of k clusters, it will need 3 * k parameters to initialize(mean, variance, scale) for each dimensiton

        """
        X_cp = self.data
        labels,centers = self.kmeans(n_cluster = clusters)
        r,c = X_cp.shape
        #### Choose pre-results from k-means
        means = centers
        cov = [np.cov(X_cp[np.where(labels == k)].T) for k in range(clusters)]
        scales = np.ones(clusters)/clusters
        delta = np.infty
        pre = 0.0
        likelihood = np.zeros([r, clusters])
        while delta > tol:
            ### E step
            for k in range(clusters):
                likelihood[:,k] = multivariate_normal( mean = means[k], cov = cov[k]).pdf(X_cp) * scales[k]

            p = likelihood/np.sum(likelihood, axis = 1).reshape(-1,1)

            logll = -np.sum(np.log(np.sum(likelihood, axis = 1)))
            delta = logll - pre
            pre = logll



            ### M-step
            ### Update parameters to maximize the log-likelihood
            for k in range(clusters):
                scales[k] = np.sum(p[:,k])/r
                means[k] = (p[:,k] @ X_cp)/np.sum(p[:,k])
                cov[k] = (p[:,k]*(X_cp - means[k]).T @ (X_cp - means[k]))/ np.sum(p[:,k])

        return np.argmax(p, axis=1)






    #############################
    ### Mean shift clustering ###
    #############################

    CLUSTER_THRESHOLD = 1e-1
    def mean_shift(self, kernel, bandwidth, tol = 1e-6):
        """
        Mean shift builds upon the concept of kernel density estimation (KDE).
        It works by placing a kernel(weighting function) on each point in the data set
        Two most used kernels are: 1. flat kernel 2. gaussian kernel
        Users define a kernel bandwidth for the kernel
        The peak of the surface for that underlying distribution defines the number of clusters to create, which is determined by the bandwidth
        the mean shift algorithm iteratively shifts each point in the data set until it the top of its nearest KDE surface peak.
        """
        X_cp = np.array(self.data)
        shift_points = [None] * X_cp.shape[0]
        shifting = [True] * X_cp.shape[0]
        while any(shifting):
            for i, point in enumerate(X_cp):
                if not shifting[i]:
                    continue
                copy_point = point.copy()
                X_cp[i] = self.shift_point(point, self.data, kernel, bandwidth)
                dist = euclidean(copy_point, point)
                if dist < tol:
                    shifting[i] = False
                    shift_points[i] = point.tolist()

        cluster_ids = self.cluster_ids(shift_points)
        return cluster_ids, np.array(shift_points)

    def shift_point(self, point, points, kernel, bandwidth):
        """
        Shift points iteratively based on weighting function windows
        """
        shift_x = 0.0
        shift_y = 0.0
        scale = 0.0
        for p in points:
            edist = euclidean(point, p)
            weight = kernel(edist, bandwidth)
            shift_x += p[0] * weight
            shift_y += p[1] * weight
            scale += weight
        shift_x = shift_x/scale
        shift_y = shift_y/scale
        return [shift_x, shift_y]

    def cluster_ids(self, points, cluster_threhold = CLUSTER_THRESHOLD):
        """
        Assign a cluster label for each data point
        Based on distance below shifted point within a certain threshold
        """
        cluster_ids = []
        centroids = []
        cluster_idx = 0
        for i, point in enumerate(points):
            if len(cluster_ids)==0:
                centroids.append(point)
                cluster_ids.append(cluster_idx)
                cluster_idx+=1
            else:
                for centroid in centroids:
                    dist = euclidean(point, centroid)
                    if dist < cluster_threhold:
                        cluster_ids.append(centroids.index(centroid))
                        break
                if len(cluster_ids) < i+1:
                    centroids.append(point)
                    cluster_ids.append(cluster_idx)
                    cluster_idx+=1
        return cluster_ids



    #############################
    ### DBSCAN clustering ###
    #############################

    def DBSCAN(self, minPoint =4, e=1):
        """
        1. Arbitrarily choose a unvisited start point and count number of neighbor points within distance \epsilon, marked that point as visited
        2. If the number of points in the neighbor exceeds certain number(minPoint), the first data point will be marked as beloinging to first cluster, so do all the neighbor points.
        Otherwise the data point will be marked as noise.
        3. Repeat step 2 for all neighbor points until all the points for the first cluster has been determined.
        4. Start with a new unvisited point and stop until membership for all data points are determined.
        """
        cluster = 1
        X_cp = self.data.copy()
        r,c = X_cp.shape
        clusters = np.zeros([r])
        visited = np.array([False] * r)
        while not all(visited):
            m = random.choice(np.where(visited == False)[0])
            initial = X_cp[m]
            visited[m] = True
            points = []
            idx = False
            points.append(initial)
            while len(points) > 0:
                cdis = cdist(X_cp, points[0].reshape(1,2))
                del points[0]
                if sum(cdis <= e) > minPoint:
                    idx = True
                    for i in range(r):
                        if cdis[i] <= e and (visited[i] == False or clusters[i] == 0):
                            clusters[i] = cluster
                            points.append(X_cp[i])
                            visited[i] = True
            if idx:
                cluster += 1
        return clusters

    #############################################
    ### Agglomerative Hierarchical clustering ###
    #############################################

    def hclust(self, endcluster):
        """
        Using average linkage
        Each data point itself is a cluster in the beginning
        Low efficiency O(n^3)
        """
        X_cp = self.data.copy()
        r,c = X_cp.shape
        ncluster = len(X_cp)
        clusters = {key: [value] for key,value in enumerate(X_cp)}
        dis = np.zeros([ncluster,ncluster])
        labels = np.empty(r)
        for c1 in clusters.keys():
            for c2 in clusters.keys():
                dis[c1, c2] = self.ave_link(clusters[c1], clusters[c2])
        while ncluster > endcluster:
            index = np.where(dis == np.min(dis[dis>0]))
            if len(index[0]) == 1:
                row,col = index[0][0],index[1][0]
            else:
                row, col = index[0][0], index[0][1]
            #### Join closest cluster
            clusters[row] = np.append(clusters[row],clusters.get(col, None),axis =0)
            del clusters[col]
            dis[col,:] = np.infty
            dis[:,col] = np.infty
            #### Update average linkage for row
            for i in clusters.keys():
                if i != row:
                    dis[row,i] = self.ave_link(clusters[row], clusters[i])
            ncluster -= 1
        ### Assign labels
        id = 0
        for c in clusters.keys():
            for val in clusters.get(c):
                ind = X_cp.tolist().index(val.tolist())
                labels[ind] = id
            id+=1
        assert len(np.unique(labels)) == endcluster
        return labels





    def ave_link(self, c1, c2):
        d = 0
        for p in c1:
            for q in c2:
                d += euclidean(p,q)
        return d/(len(c1) * len(c2))


def euclidean(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))

def gaussian_kernel(distance, bandwidth):
    return 1 / (bandwidth * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (distance / bandwidth) ** 2)

def flat_kernel(distance, bandwidth):
    if distance > bandwidth:
        return 0
    else:
        return 1

def main():
    npts = 1000
    nlbl = 6
    np.random.seed(2020)
    X,y = make_blobs(n_samples=npts,centers=nlbl)
    #from sklearn.datasets import make_circles
    #X, y = make_circles(factor=0.5, random_state=0, noise=0.05)
    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.get_cmap('Accent', nlbl))
    plt.savefig('./figures/origin.png')
    sample = runClusters(data = X, label=y)
    b_label, b_center = sample.kmeans(n_cluster=nlbl)
    plt.figure(2)
    plt.scatter(X[:,0], X[:, 1], s=40, c=b_label, cmap=plt.cm.get_cmap('Accent', nlbl))
    plt.scatter(b_center[:,0],b_center[:,1], marker = 'x', s = 100 ,c ='red', linewidth = 2)
    plt.savefig('./figures/kmeans.png')
    c_label, c_center = sample.k_means_pp(n_clusters=nlbl)
    plt.figure(3)
    plt.scatter(X[:,0], X[:,1], c = c_label, s=40, cmap = plt.cm.get_cmap('Accent', nlbl))
    plt.scatter(c_center[:,0],c_center[:,1], marker = 'x', s = 100 ,c ='red', linewidth = 2)
    plt.savefig('./figures/kmeans_plus.png')
    d_label, d_peaks =  sample.mean_shift(kernel=gaussian_kernel, bandwidth = 1)
    plt.figure(4)
    plt.scatter(X[:,0],X[:,1], s = 40, c = d_label, cmap=plt.cm.get_cmap('Accent', len(np.unique(c_label))))
    plt.scatter(d_peaks[:,0], d_peaks[:,1], c = 'red', s = 80)
    plt.savefig('./figures/mean_shift.png')
    e_label = sample.DBSCAN(minPoint=10, e=2)
    plt.figure(5)
    plt.scatter(X[:,0],X[:,1], s =40, c = e_label, cmap=plt.cm.get_cmap('Accent',len(np.unique(e_label))))
    plt.savefig('./figures/dbscan.png')
    f_label = sample.gaussian_mix(clusters=nlbl, tol = 1e-3)
    plt.figure(6)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=f_label, cmap=plt.cm.get_cmap('Accent', nlbl))
    plt.savefig('./figures/GMM.png')
    g_label = sample.hclust(endcluster=nlbl)
    plt.figure(7)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=g_label, cmap=plt.cm.get_cmap('Accent', nlbl))
    plt.savefig('./figures/hclust.png')

    plt.show()
if __name__ == "__main__":
    main()