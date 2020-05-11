#!/usr/bin/python3
import numpy as np
import itertools as itr
import functools as ft

# NOTE: Branching factor cannot be 1 and possibly not 2 either, for performance reasons?
# TODO: Create a distance metric class containing methods that correspond to the
#       different distance metrics in the BIRCH paper, the CF tree will store a chosen method/metric
#       as an attribute


class ClusterFeature(object):
    """Cluster Feature object.

    Stores the details of a cluster/sub-cluster.

    Attributes:
        n (int): The number of data points in the cluster.
        ls ([float]): NOTE: This is represented using a numpy array. The linear
        sum of the n data points, a vector of size d. Where d is the number
        of features to cluster the data-set on.
        ss (float): The square sum of the n data points.
        timestamps ([datetime]): An arrary of size n_taxis, each element
        containing the timestamps at which this cluster was visited. The index represents the taxi id.
    """

    def __init__(self, n, ls, ss):
        self.n = n
        self.ls = ls
        self.ss = ss
        self.timestamps = []

    def __add__(self, other):
        return ClusterFeature(self.n + other.n, self.ls + other.ls, self.ss + other.ss)

    def __iadd__(self, other):
        self.n += other.n
        self.ls += other.ls
        self.ss += other.ss
        return self

    def __eq__(self, other):
        if self.n == other.n and (self.ls == other.ls).all() and \
                self.ss == other.ss and self.timestamps == other.timestamps:
            return True
        return False

    def centroid(self):
        return self.ls / self.n

    def radius(self):
        """
        The radius is the average distance from member points to the centroid.
        """
        wiki = self.ss / self.n - self.centroid()**2
        c = self.centroid()
        mine = (self.n*c**2 + self.ss - 2*c*self.ls)/self.n
        #return self.centroid().dot(self.centroid())
        #return self.ss / self.n, self.centroid()**2
        return np.sqrt(self.ss / self.n - c.dot(c))

    def distance_metric(self, other):
        """
        A Euclidian distance metric. Referred to as D0 in the BIRCH clustering paper.
        """
        return np.sqrt((self.centroid() - other.centroid())**2)

    def show(self):
        print("N: {}, LS: {}, SS: {}, Radius: {}, Centroid: {}".format(
            self.n, self.ls, self.ss, self.radius(), self.centroid()))


class Node(object):
    """Tree node object.

    Each node stores Cluster Features and its children. Children are unique to each cluster.

    Attributes:
        order (int): Branching factor. The maximum number of Cluster Features each node can hold.
        cluster_features ([ClusterFeature]): A list containing the CF(s) within the node.
        children ([Node]): Contains pointers to the children in the case that it is not a leaf node.
        If the node is a leaf node, then the list is empty.
    """

    def __init__(self, order=8, feature_children=None):
        self.order = order
        if feature_children is None:
            self.cluster_features = []
            self.children = []
        else:
            self.cluster_features = feature_children[0]
            self.children = feature_children[1]

    # could be merged with find node function inside CFTree
    # (self,entry,type) type==c - cluster type==n - node

    def _find(self, entry, type):
        """Finds the cloest cluster to the entry.

        Determines the closest cluster relative to the entry cluster, within the node.

        Args:
            self (Node): A Node instance
            entry (ClusterFeature): A cluster feature that is to be added to the tree.
            type (char): Determines what is returned by the function. 'c' indicates that
            a cluster is to be returned. 'n' indicates that a node (child) is to be
            returned.

        Returns:
            A cluster followed by its position in the node. OR
            A child entry followed by its position in the node.
        """
        (index, distance) = (-1, np.inf)
        for i, item in enumerate(self.cluster_features):
            centroid_distance = item.distance_metric(entry)
            if centroid_distance < distance:
                index = i
                distance = centroid_distance
        if type == 'c':
            return self.cluster_features[index], index
        else:
            return self.children[index], index

    def add_entry(self, entry, threshold):
        current_cf, index = self._find(entry, 'c')
        new_cf = current_cf + entry
        if new_cf.diameter() < threshold:
            self.cluster_features[index] = new_cf
        else:
            self.cluster_features.append(entry)

    def seed_split(self):
        """Splits the node into two nodes.

        It choses the two clusters that are furthest apart, with regard to the chosen distance
        metric. These are the seeds, the two seeds are placed into seperate new nodes. The remaining
        entries are distributed between these nodes, based on the which cluster they are closest to.
        """

        (seed_l, seed_r, max_dist) = (None, None, np.NINF)
        for x, y in itr.permutations(self.cluster_features, r=2):
            if x.distance_metric(y) > max_dist:
                seed_l = x
                seed_r = y
        left = Node(self.order, ([seed_l], []))
        right = Node(self.order, ([seed_r], []))

        # when storing the max distance, also store the index
        # pop the entry from the list
        # then the first if statement can be removed
        # pop each element as we go?
        for i, cf in self.cluster_features:
            if cf != seed_l and cf != seed_r:
                if cf.distance_metric(seed_l) < cf.distance_metric(seed_r):
                    left.add_entry(cf)
                else:
                    right.add_entry(cf)
        return left, right

    def _summarise(self):
        """
        Creates a summary Cluster Feature of the Cluster Features within the node.
        """
        return sum(cf for cf in self.cluster_features)

    def is_full(self):
        return len(self.cluster_features) == self.order


class CFTree(object):
    """
    Clustering Feature tree object, consisting of nodes.

    Nodes will automatically be split into two when the number of cluster features it contains,
    is greater than the branching factor. If the root is split, the height increases by one.

    When a split occurs, summaries (Cluster Features) of all sub Cluster Features within each
    node will be created and subsequently inserted into the parent node. If the
    parent is full (number of entries is greater than the branching factor) we must split it.
    This principle is applied recursively until we find space for the node.
    i.e. If the grand-parent node is also full.

    After this, we must update the CFs on the path from the leaf to the root.

    Attributes:
        root (Node): The root of the tree.
        order (int): The maximum number of keys each node can hold.
        threshold (float): The radius of each cluster must be less than this value.
    """

    def __init__(self, order=8, threshold=0.5):
        self.root = Node(order)
        self.order = order
        self.threshold = threshold

    def _split_summarise(self, node):
        left, right = node.seed_split()
        left_summary = left._summarise()
        right_summary = right._summarise()
        return ((left_summary, right_summary), (left, right))

    def _merge(self, parent, index, cfs, children):
        # CFs within a node are not sorted as there is no condition to sort by
        # that would speed up insertions and retrievals

        parent.cluster_features[index] = cfs[0]
        parent.children[index] = children[0]

        parent.cluster_features = parent.cluster_features[:index] + \
            cfs[1] + parent.cluster_features[index:]
        parent.children[index] = parent.children[:index] + \
            children[1] + parent.children[index:]

    def insert_point(self, X):
        """Inserts a point after traversing to a leaf node.

        If the leaf node is full, split the leaf node into two, and add to the parent node.
        Repeat up to the first ancestor with free space.

        Args:
            self (CFTree): A Cluster Feature tree instance.
            X (Vector): A vector/point that is to be added to a cluster in the tree.
        """

        parents = [(None, None)]
        child = self.root
        entry_cluster = ClusterFeature(n=1, ls=sum(
            X), ss=ft.reduce(lambda x, y: x + y * y, X))

        while child.children != []:
            parent = child
            child, index = child._find(entry_cluster, 'n')
            parents.append((parent, index))

        child.add_entry(entry_cluster)

        # index contains the position (index) of the child node in the parent
        # this must be modified to now represent one of the split nodes,
        # and another cf must be inserted into the parent to represent the another
        # split node

        # reverse parents list

        # not sure this is sound syntax
        for parent, index in parents.reverse():
            if child.is_full():
                # we must split
                if parent is None:
                    # create new root
                    self.root = Node(
                        order=self.order, feature_children=self._split_summarise(child))
                    return
                else:
                    # we can insert the summaries into the parent (in the next iteration we check if that insertion made the parent full)
                    self._merge(parent, index, self._split_summarise(child))
                    child = parent
            else:
                # no spliting, but updating the path from leaf to root
                if parent is None:
                    return
                else:
                    #self._merge(parent, index, _summarise(child))
                    # must update all ancestors of the child
                    parent.cluster_features[index] = child._summarise()
                    child = parent

        # refinement step done here


def test_module():
    points = np.array([[2, 3], [2, 2], [1, 3], [10, 11], [11, 11], [10, 12]])
    print(np.square(points[0]))
    cf = ClusterFeature(n=1, ls=points[0], ss=np.sum(points[0]**2))
    cf_2 = ClusterFeature(n=1, ls=points[1], ss=np.sum(points[1]**2))
    cf.show()
    cf_2.show()
    cf_3 = cf + cf_2
    cf_3.show()
    cf += cf_2
    cf.show()
    assert cf == cf_3


if __name__ == '__main__':
    test_module()
