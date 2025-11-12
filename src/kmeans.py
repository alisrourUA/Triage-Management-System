# here we did the k-means clustering for k=3

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def compute_centroid(points):
    if not points:
        return (0, 0)
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def kmeans(points, k=3, max_iter=100):
    centroids = points[:k]
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]

        for p in points:
            distances = [euclidean_distance(p, c) for c in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(p)

        new_centroids = [compute_centroid(cluster) for cluster in clusters]

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids, clusters
