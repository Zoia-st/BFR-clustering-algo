import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
np.seterr(divide='ignore', invalid='ignore')

st.title("Data analysis with the BFR algorithm")
st.sidebar.title("Choose parameters")

input_filename = st.sidebar.text_input('Enter a path to input filename (e.g. /Users/Xiaomi/Desktop/...)',
                                       '/Users/Xiaomi/Desktop/Clustering project/BFR_input.csv')
n_cluster = st.sidebar.slider("The number of clusters", min_value=2, max_value=10, value=3)
chunks = st.sidebar.slider("The number of chunks", min_value=2, max_value=10, value=3)
output_filename = st.sidebar.text_input('Enter a path to output filename (e.g. /Users/Xiaomi/Desktop/...)',
                                        '/Users/Xiaomi/Desktop/Clustering project/BFR_results.txt')


def read_file(input_filename):
    """ Reads specified file, adds points in array, mixes points in array

        and divides them into chunks. """

    file = open(input_filename, 'r')
    data = list()
    for line in file:
        data.append([float(i.strip()) for i in line.split(',')])
    data = np.array(data)
    #np.random.shuffle(data)
    data_parts = np.array_split(data, chunks)
    return data_parts


def group_cluster(labels, data):
    """ Creates dictionary with clusters as keys

        and assigned to them points. """

    assert len(labels) == data.shape[0]
    idx = np.arange(data.shape[0])
    clusters_dict = dict()
    for l, i in zip(labels, idx):
        if l in clusters_dict:
            clusters_dict[l].append(i)
        else:
            clusters_dict[l] = [i]

    return clusters_dict


def process_cluster(cluster_dict):
    """ Goes through all clusters and returns two lists with clusters.

        Others - points of clusters with multiple points in one cluster. """

    single_cluster_points, others = list(), list()
    for k, v in cluster_dict.items():
        if len(v) == 1:
            single_cluster_points.extend(v)
        else:
            others.extend(v)
    return single_cluster_points, others


def group_process_cluster(labels, idx):
    """ Combines functions def group_cluster(labels, data)

        and def process_cluster(cluster_dict). """

    cluster_dict = group_cluster(labels, idx)
    return process_cluster(cluster_dict)


def summarise_clusters(data, labels):
    """ Summarizes information about clusters: N, SUM and SUMSQ. """

    assert data.shape[0] == len(labels)
    compressed_clusters = []
    cluster_dict = group_cluster(labels, data)

    for k, v in cluster_dict.items():
        compressed_clusters.append({
            'n': len(v),
            'sum': data[v, :].sum(axis=0),
            'sumsq': np.square(data[v, :]).sum(axis=0)
        })

    return compressed_clusters


def cluster_kmeans(n_cluster, data):
    """ Runs function kmean from the library Sklearn. """

    return KMeans(n_clusters=n_cluster).fit(data)


def mahalanobis(x, c, sigma):
    """ Returns the calculated Mahalanobis distance. """

    return np.sqrt(np.sum(np.square((x - c) / sigma)))


def mahalanobis_distance(cluster, point):
    """ Calculate centroid and sigma of the cluster. """

    center = cluster['sum'] / cluster['n']
    sigma = np.sqrt((cluster['sumsq'] / cluster['n']) - np.square(center))
    return mahalanobis(point, center, sigma)


def add_point_cluster(point, cluster):
    """ Updates the information about cluster regarding to assigned points. """

    cluster['n'] += 1
    cluster['sum'] += point
    cluster['sumsq'] += np.square(point)
    return cluster


def add_to_clusters(points, clusters):
    """ Adds points to the nearest cluster based on the Mahalanobis distance

        calculation, which should be less than threshold (2√d). """

    threshold = 2 * np.sqrt(points.shape[1])
    unassigned = list()
    for p in range(len(points)):
        min_d, assignment = None, None
        for i in range(len(clusters)):
            d = mahalanobis_distance(clusters[i], points[p])
            if min_d is None:
                min_d, assignment = d, i
                continue
            if d < min_d:
                min_d, assignment = d, i
                
        if min_d < threshold:
            clusters[assignment] = add_point_cluster(points[p], clusters[assignment])
        else:
            unassigned.append(p)
    return unassigned


def distance_bw_clusters(c1, c2):
    """ Calculates the Mahalanobis distance between clusters. """

    center1 = c1['sum'] / c1['n']
    center2 = c2['sum'] / c2['n']
    sd1 = np.sqrt((c1['sumsq'] / c1['n']) - np.square(center1))
    sd2 = np.sqrt((c2['sumsq'] / c2['n']) - np.square(center2))
    return mahalanobis(center1, center2, sd2 * sd1)


def join_clusters(c1, c2):
    """ Updates information about merged slusters (N, SUM, SUMSQ). """

    c1['n'] += c2['n']
    c1['sum'] += c2['sum']
    c1['sumsq'] += c2['sumsq']
    return c1


def count_points(cluster_dicts):
    """ Sums all points in discard and in compressed sets. """

    return sum([cluster_dict['n'] for cluster_dict in cluster_dicts])


def merge_clusters(old_clusters, new_clusters, threshold, return_two):
    """ Merges compressed sets (or discard and compressed sets)

        based on the Mahalanobis distance."""

    merge_tuples = list()
    for j in range(len(new_clusters)):
        min_d, assignment = None, None
        for i in range(len(old_clusters)):
            d = distance_bw_clusters(old_clusters[i], new_clusters[j])
            if min_d is None:
                min_d, assignment = d, i
                continue
            if d < min_d:
                min_d, assignment = d, i
        if min_d < threshold:
            merge_tuples.append((assignment, j))
        else:
            continue

    for i, j in merge_tuples:
        old_clusters[i] = join_clusters(old_clusters[i], new_clusters[j])

    for i, j in sorted(merge_tuples, key=lambda x: x[1], reverse=True):
        new_clusters.pop(j)

    if return_two:
        return old_clusters, new_clusters
    else:
        old_clusters.extend(new_clusters)
        return old_clusters


def print_stats(ds_clusters, cs_clusters, rs_points):
    """ Prints current information about clusters and points. """

    stats = generate_stats(ds_clusters, cs_clusters, rs_points)
    stats['SUM'] = stats['DS Points'] + stats['CS Points'] + stats['RS Points']
    print(stats)
    return stats


def generate_stats(ds_clusters, cs_clusters, rs_points):
    """ Generates status with the number of discard and compressed sets

        (clusters) and the number of points in discard,

        compressed and retained sets. """

    return {'DS Clusters': len(ds_clusters),
        'DS Points': count_points(ds_clusters),
        'CS Clusters': len(cs_clusters),
        'CS Points': count_points(cs_clusters),
        'RS Points': len(rs_points)}


def predict(clusters, data, threshold):
    """ Assigns every point to the cluster, if a point can’t be assigned

        to any cluster, it gets value -1. """

    final_labels = []
    for chunk in range(chunks):
        for p in range(len(data[chunk])):
            min_d, assignment = None, None
            for c in range(len(clusters)):
                d = mahalanobis_distance(clusters[c], data[chunk][p, 1:])
                if min_d is None:
                    min_d, assignment = d, c
                    continue
                if d < min_d:
                    min_d, assignment = d, c

            if min_d < threshold:
                final_labels.append((int(data[chunk][p, 0]), assignment))
            else:
                final_labels.append((int(data[chunk][p, 0]), -1))
    final_labels = sorted(final_labels, key=lambda x: x[0])
    return final_labels


def write_output(stats, predictions):
    """ Prints the final results in specified file, all information about sets

        (clusters) and points can be found in the file. """

    st.subheader('The intermediate results:\n')
    idx = 1
    for i in stats:
        st.write('Round ' + str(idx) + ': ' +
                   'DS Points: ' + str(i['DS Points']) + ',' +
                   'CS Clusters: ' + str(i['CS Clusters']) + ',' +
                   'CS Points: ' + str(i['CS Points']) + ',' +
                   'RS Points: ' + str(i['RS Points']) + '\n')
        idx += 1
    st.subheader('Clustering results:')
    st.write('See specified output file.')
    file = open(output_filename, 'w')
    for idx, val in predictions:
        file.write(str(idx) + ',' + str(val) + '\n')
    file.close()


def main():
    # Step 1 : Load data into 5 chunks
    data = read_file(input_filename)
    threshold = 2 * np.sqrt(data[0].shape[1] - 1)
    stats = []
    cs_clusters, new_cs_clusters = [], []
    rs_points = None

    # Step 2 : Run K-Means for the first part of dataset
    kmeans = cluster_kmeans(n_cluster, data[0][:, 1:])

    df = data[0][:, 1:]
    kmeans1 = KMeans(n_clusters=n_cluster)
    label = kmeans1.fit_predict(df)
    centroids = kmeans1.cluster_centers_
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i, 0], df[label == i, 1], label=i)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
    plt.legend()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()  # cluster and centroids visualization
    st.write('The plot shows clusters and centroids (black points) for the first part of data. '
             'The next data loads will be distributed according to these clusters.')

    # Step 3 : Summarize information about DS set
    ds_clusters = summarise_clusters(data[0][:, 1:], kmeans.labels_)

    # Step 4: load next chunk and assign new points to nearest DS clusters and filter unassigned
    for i in range(1, chunks):

        unassigned = add_to_clusters(data[i][:, 1:], ds_clusters)

        # Step 5: Unassigned points are added to nearest CS clusters
        if len(cs_clusters) > 0:
            if len(unassigned):
                unassigned = add_to_clusters(data[i][unassigned, 1:], cs_clusters)

        # Step 6: Unassigned points are added to rs
        if len(unassigned) > 0:
            if i == 1:
                rs_points = data[i][unassigned, 1:]
            else:
                rs_points = np.append(rs_points, data[i][unassigned, 1:], axis=0)

        # Step 7: Run kmeans on RS points and create CS cluster and RS points
        if len(rs_points) >= 5 * n_cluster:
            kmeans = cluster_kmeans(5 * n_cluster, rs_points)
            rs_idx, cs_idx = group_process_cluster(kmeans.labels_, rs_points)
            if len(cs_idx) > 0:
                if len(cs_clusters) > 0:
                    new_cs_clusters = summarise_clusters(rs_points[cs_idx], kmeans.labels_[cs_idx])
                else:
                    cs_clusters = summarise_clusters(rs_points[cs_idx], kmeans.labels_[cs_idx])
                if len(new_cs_clusters):
                    cs_clusters = merge_clusters(cs_clusters, new_cs_clusters, threshold, return_two=False)

            if len(rs_idx) > 0:
                rs_points = rs_points[rs_idx]

        if i < (chunks - 1):
            stats.append(print_stats(ds_clusters, cs_clusters, rs_points))

    # Step 8: Merge cs and ds clusters
    ds_clusters, cs_clusters = merge_clusters(ds_clusters, cs_clusters, threshold, return_two=True)
    stats.append(print_stats(ds_clusters, cs_clusters, rs_points))

    #  Step 9: Generate predictions
    predictions = predict(ds_clusters, data, threshold)
    write_output(stats, predictions)

    #  Algorithms evaluation

    with open(output_filename, 'r') as csv_file:
        lines = csv_file.readlines()
    clustersBFR = []
    for line in lines:
        data = line.split(',')
        clustersBFR.append(int(data[1]))  #  list with clusters from BFR algo

    df = []
    data = pd.read_csv(input_filename, header=None, prefix='var')
    data = data.drop(data.columns[0], axis=1)
    df = df.append(data)
    df = np.array(data)

    #Hierarchical clustering
    cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df)
    clustersHC = list(cluster.labels_)


    st.subheader('Comparison of the algorithms:')
    st.write('The Rand index (BFR and Hierarchical clustering):', adjusted_rand_score(clustersBFR, clustersHC))

    #Kmean clustering
    cluster = KMeans(n_clusters=n_cluster)
    cluster_labels = cluster.fit_predict(df)
    clustersK = list(cluster_labels)

    st.write('The Rand index (BFR and Kmean clustering):', adjusted_rand_score(clustersBFR, clustersK))


if __name__ == '__main__':
    main()
