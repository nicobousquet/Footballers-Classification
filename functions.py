# file functions.py
# Nicolas Bousquet
import itertools
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import parallel_coordinates
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import linalg
import matplotlib as mpl

palette = sns.color_palette("bright", 10)


def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    '''Display a scatter plot on a factorial plane, one for each factorial plane'''

    # For each factorial plane
    for d1, d2, d3 in axis_ranks:
        if d3 < n_comp:

            # Initialise the matplotlib figure
            fig = plt.figure(figsize=(7, 6))

            # Display the points
            ax = plt.axes(projection="3d")
            if illustrative_var is None:
                ax.scatter3D(X_projected[:, d1], X_projected[:, d2], X_projected[:, d3], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    ax.scatter3D(X_projected[selected, d1], X_projected[selected, d2], X_projected[selected, d3],
                                 alpha=alpha, label=value)
                plt.legend()

            # Display the labels on the points
            if labels is not None:
                for i, (x, y, z) in enumerate(X_projected[:, [d1, d2, d3]]):
                    plt.text(x, y, z, labels[i],
                             fontsize='14', ha='center', va='center')

                    # Define the limits of the chart
            boundary = np.max(np.abs(X_projected[:, [d1, d2, d3]])) * 1.1
            ax.set_xlim([-boundary, boundary])
            ax.set_ylim([-boundary, boundary])
            ax.set_zlim([-boundary, boundary])

            # Display grid lines
            plt.plot([-100, 100], [0, 0], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [0, 0], [-100, 100], color='grey', ls='--')

            # Label the axes, with the percentage of variance explained
            ax.set_xlabel('PC{} ({}%)'.format(d1 + 1, round(100 * pca.explained_variance_ratio_[d1], 1)))
            ax.set_ylabel('PC{} ({}%)'.format(d2 + 1, round(100 * pca.explained_variance_ratio_[d2], 1)))
            ax.set_zlabel('PC{} ({}%)'.format(d3 + 1, round(100 * pca.explained_variance_ratio_[d3], 1)))

            plt.title("Projection of points (on PC{} and PC{} and PC{}])".format(d1 + 1, d2 + 1, d3 + 1))
            # plt.show(block=False)
    return ax


# def display_scree_plot(pca):
#     '''Display a scree plot for the pca'''
#
#     scree = pca.explained_variance_ratio_ * 100
#     plt.bar(np.arange(len(scree)) + 1, scree)
#     plt.plot(np.arange(len(scree)) + 1, scree.cumsum(), c="red", marker='o')
#     plt.xlabel("Number of principal components")
#     plt.ylabel("Percentage explained variance")
#     plt.title("Scree plot")
#     plt.show(block=False)

# def addAlpha(colour, alpha):
#     '''Add an alpha to the RGB colour'''
#
#     return (colour[0], colour[1], colour[2], alpha)
#

# def display_parallel_coordinates(df, num_clusters):
#     '''Display a parallel coordinates plot for the clusters in df'''
#
#     # Select data points for individual clusters
#     cluster_points = []
#     for i in range(num_clusters):
#         cluster_points.append(df[df.cluster == i])
#
#     # Create the plot
#     fig = plt.figure(figsize=(12, 15))
#     title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
#     fig.subplots_adjust(top=0.95, wspace=0)
#
#     # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other
#     # clusters
#     for i in range(num_clusters):
#         plt.subplot(num_clusters, 1, i + 1)
#         for j, c in enumerate(cluster_points):
#             if i != j:
#                 pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j], 0.2)])
#         pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i], 0.5)])
#
#         # Stagger the axes
#         ax = plt.gca()
#         for tick in ax.xaxis.get_major_ticks()[1::2]:
#             tick.set_pad(20)


def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)


color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange", "r", "g", "b", "y"])


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(1, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")


# Generate some test data
data = np.arange(200).reshape((4, 5, 10))


def export_3D_array(filename, array):
    # Write the array to disk
    with open(filename, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(array.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in array:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f', delimiter=',')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
