from scipy.spatial.distance import pdist


class DBSCAN(object):
    def __init__(self, eps, min_neighbors, metric='euclidean'):
        """
        A re-creation of the DBSCAN algorithm, using only scipy as a dependency.

        :param eps: Epsilon, aka the distance between two points to be considered 'neighbors'.
        :param min_neighbors: The number of neighbors (including itself) a point must have to be labeled as a core
        point.
        :param metric: The distance metric to be used. Default is 'euclidean'.
        The distance metric can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, 
        ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, 
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
        ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, or ‘yule’.
        """
        self.labels = []
        self.eps = eps
        self.min_neighbors = min_neighbors
        self.metric = metric
        self.num_points = 0
        self.distances = []

    def __find_neighbors(self, point_index):
        """
        Returns a list containing the indexes of points that are within epsilon distance of a given data point.

        :param point_index: Index of data point to check for neighbors.
        :return: List of index integers corresponding to neighbors of the given point.
        """
        neighbors_indexes = set()

        for i in range(self.num_points):
            if i != point_index and self.__get_dist(point_index, i) <= self.eps:
                neighbors_indexes.add(i)

        return neighbors_indexes

    def __get_dist(self, index_1, index_2):
        """
        Function to fetch the right data from the list self.distances given the indexes of two data points.

        :param index_1: Index of first data point
        :param index_2: Index of second data point
        :return: Distance between the two points
        """
        if index_1 == index_2:  # Really shouldn't be equal, but just in case...
            return 0

        if index_1 < index_2:  # Make index 1 bigger, if it's not already.
            index_1, index_2 = index_2, index_1

        return self.distances[self.num_points * index_2 - index_2 * (index_2 + 1) // 2 + index_1 - 1 - index_2]

    def fit(self, data):
        """
        Apply a 2 by n data set, where n is the number of points, applying labels to each point.
        :param data: A 2 by n 2D data set. Ex: [ [1,2], [2,3], [3,4] ]
        :return: A list of n length, containing labels in the form of integers. -1 is noise, 0 and up are groups.
        """
        self.num_points = len(data)
        self.labels = [None] * self.num_points

        # The pdist function is apparently an efficient way to calculate the distance between every point!
        self.distances = pdist(data, metric=self.metric)

        self.calculate_labels()  # Label calculation Given its own method

    def calculate_labels(self):
        """
        Method that calculates all the labels in the self.labels list, using current distance data and epsilon value.
        """
        current_label = -1
        for index in range(self.num_points):
            if self.labels[index] is not None:  # Skip any previously processed points.
                continue

            neighbors_indexes = self.__find_neighbors(index)  # Get set of neighbors in point

            if len(neighbors_indexes) + 1 < self.min_neighbors:  # If not enough neighbors, label as noise.
                self.labels[index] = -1
                continue

            current_label += 1  # If we get this far, we've found a new cluster. Move on to next label.
            self.labels[index] = current_label  # Label 'core' point

            already_processed = {index}  # A set used to keep track of what we've already checked in the next step.

            while neighbors_indexes:
                # Pick out a neighbor from the queue and add it to the already_processed set.
                neighbor_index = neighbors_indexes.pop()
                already_processed.add(neighbor_index)

                # Change neighbors previously labeled as noise to border points.
                if self.labels[neighbor_index] == -1:
                    self.labels[neighbor_index] = current_label

                # Skip any previously processed points, or any we just labeled as border points.
                if self.labels[neighbor_index] is not None:
                    continue

                # Label neighbor
                self.labels[neighbor_index] = current_label

                # Find all neighbors of this neighbor, and add any that qualify as a 'core' point to the queue.
                more_neighbors = self.__find_neighbors(neighbor_index)
                if len(more_neighbors) >= self.min_neighbors:  # If it has enough neighbors, add to the neighbor set
                    for neighbor in more_neighbors:
                        if neighbor not in already_processed:
                            neighbors_indexes.add(neighbor)

        # End of algorithm!

    def predict(self):
        return self.labels.copy()


class KDBSCAN(DBSCAN):
    def __init__(self, k, min_neighbors, metric='euclidean', max_tries=20):
        """
        An algorithm similar to DBSCAN, except it takes an input of K instead of epsilon. It then uses a binary search
        pattern to find a value of epsilon that results in K clusters.

        The distance metric can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’,
        ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
        ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’,
        ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, or ‘yule’.

        :param k: The number of clusters to aim for in the final result.
        :param min_neighbors: The number of neighbors a point must have to start a cluster.
        :param metric: The metric to be used in the distance calculation. Default is 'euclidean'.
        :param max_tries: Number of passes the algorithm should make at calculating epsilon before quitting.
        """
        self.labels = []
        self.k = k  # Number of groups to be made
        self.max_tries = max_tries  # This will be useful for avoiding an endless loop.
        self.eps = 0  # Epsilon, aka the minimum distance between two points to be 'neighbors'.
        self.min_neighbors = min_neighbors
        self.metric = metric
        self.num_points = 0
        self.distances = []

    def fit(self, data):
        self.num_points = len(data)

        # The pdist function is apparently an efficient way to calculate the distance between every point!
        self.distances = pdist(data, metric=self.metric)

        # Use binary search to find optimal epsilon value!
        low = min(self.distances)
        high = max(self.distances)
        self.eps = (low + high) / 2
        num_tries = 0

        while num_tries <= self.max_tries:
            num_tries =+ 1
            self.labels = [None] * self.num_points

            # Calculate the labels to see how many we end up with using this epsilon value!
            self.calculate_labels()

            num_groups = max(self.labels) + 1
            if num_groups < self.k:
                # Epsilon too big, guess smaller next pass!
                high = self.eps
                self.eps = (low + high) / 2
            elif num_groups > self.k:
                # Epsilon too small, guess bigger next pass!
                low = self.eps
                self.eps = (low + high) / 2
            else:
                # Just right!
                break

        # End of algorithm!


if __name__ == "__main__":
    X = [
        [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]
    ]
    clustering = DBSCAN(eps=3, min_neighbors=2)
    clustering.fit(X)
    labels = clustering.predict()

    print(labels)

    clustering = KDBSCAN(k=2, min_neighbors=2)
    clustering.fit(X)
    labels = clustering.predict()

    print(labels)
