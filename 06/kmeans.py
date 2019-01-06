import numpy as np
import sys
import matplotlib.pyplot as plt

############################################################
#
#                       KMEANS
#
############################################################

plt.axis('equal')
plt.ion()

num_clusters = 4
num_iter = 20

# TODO load 2D features
# points = [(x, y) for x, y in [z.split() for z in open("kmeans_points.txt", "r").readlines()]]
points = np.loadtxt("kmeans_points.txt")

cluster_colors = ['r', 'b', 'y', 'g', 'm', 'k']
current_cluster_centers = np.zeros((num_clusters, 2), np.float32)
cluster_indices = np.zeros(points.shape[0], dtype=np.uint8)


def distance(a, b):
    return np.linalg.norm(a - b)


def update_mean():
    """
    Updates the cluster centers current_cluster_centers[i] depending on its new point assignment.
    This is the mean of all points belonging to this cluster center.
    :return: nothing
    """
    # TODO: implement update cluster center means
    assert len(points) == len(cluster_indices)

    for i in range(num_clusters):
        filtered = np.argwhere(cluster_indices == i)
        values = np.take(points, filtered, axis=0)
        current_cluster_centers[i] = np.mean(values, axis=0)


def assign_to_current_mean():
    """
    Assign each data point to its nearest cluster center by
    computing its eucledian distance. The function returns the overall distance
    as a measure of global change (assignment switch from from one cluster to another
    would also be a good indicator)

    :return: overall distance
    """

    # TODO: implement assignment of each point to its cluster center
    overall_dist = 0
    for i, point in enumerate(points):
        distance = np.PINF
        cluster_index = 0
        for j, cluster_center in enumerate(current_cluster_centers):
            temp_dist = np.linalg.norm(point - cluster_center)
            if temp_dist < distance:
                distance = temp_dist
                cluster_index = j
        cluster_indices[i] = cluster_index
        overall_dist += distance

    # TODO: save cluster assignment for each point in cluster_indices (this should correspond to the order of the points)
    # e.g. cluster_indices[i] = clusterindex for the ith point in points

    # YOUR CODE HERE
    return overall_dist


def initialize():
    """
    Initialize k-means randomly by assigning num_clusters cluster centers
    :return:
    """

    # Generiere zufällige Punkte mit Gaußscher Normalverteilung
    # Mittelwert aller Punkte finden
    mean = np.mean(points, axis=0)
    # Standardabweichung aller Punkte finden
    std = np.std(points, axis=0)

    # Zufällige Punkte erzeugen
    current_cluster_centers[:] = np.random.normal(mean, std, (num_clusters, 2))

    # TODO: implement random cluster assignment
    # for i in range(num_clusters):
    #     current_cluster_centers[i] = (
    #         np.random.uniform(points[:, 1].min(), points[:, 1].max()),
    #         np.random.uniform(points[:, 0].min(), points[:, 0].max())
    #     )


if __name__ == "__main__":

    # TODO initialize
    dist = sys.float_info.max
    initialize()

    for r in range(0, num_iter):

        plt.clf()

        # TODO implement assignment
        new_dist = assign_to_current_mean()
        # TODO implement udpate mean
        update_mean()

        # TODO stop iteration when overall distance aka rate of change is 'reasonably' small or
        # number of iterations exceeds maximum
        if new_dist <= dist:
            diff = dist - new_dist
            print(new_dist, diff)
            if diff <= 0.01 * new_dist:
                print("diff below 1%. exiting...")
                break
            dist = new_dist

        # Plotting the clusters
        clusters = []
        for i in range(num_clusters):
            indices = np.where(cluster_indices == i)[0]
            clusters.append(points[indices])
            plt.plot(*zip(*points[indices]), marker='o', color=cluster_colors[i], ls='')

        # Plotting cluster centers
        for i in range(num_clusters):
            plt.plot(current_cluster_centers[i][0], current_cluster_centers[i]
                     [1], marker=r'$\bowtie$', color='k', markersize=10)

        # Save cluster and pause / uncomment if needed
        # plt.savefig('cluster_'+str(r)+'.png')
        # pause in seconds
        plt.pause(1)
