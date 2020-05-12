#!/usr/bin/python3
import numpy as np
import itertools as itr
import functools as ft

# NOTE: Branching factor cannot be 1 and possibly not 2 either, for performance reasons?
# TODO: Create a distance metric class containing methods that correspond to the
#       different distance metrics in the BIRCH paper, the CF tree will store a chosen method/metric
#       as an attribute
# TODO:
#       1) Adapt CFtree and Node class to use np.array for ls.
#       2) Carry on with testing the module.


class ClusterFeature(object):
    """Cluster Feature object.

    Stores the details of a cluster/sub-cluster.

    Attributes:
        n (int): The number of data points in the cluster.
        ls ([float]): NOTE: This is represented using a numpy array. The linear
        sum of the n data points, a vector of size d. Where d is the number
        of features to cluster the data-set on.
        ss (float): The square sum of the n data points.
        timestamps ([[datetime]]): An arrary of size n_taxis, each element
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

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

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
        c = self.centroid()
        return np.sqrt(self.ss / self.n - c.dot(c))

    def distance_metric(self, other):
        """
        A Euclidian distance metric. Referred to as D0 in the BIRCH clustering paper.
        """
        dif = self.centroid() - other.centroid()
        return np.sqrt(dif.dot(dif))

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
        index = -1
        distance = np.inf
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
        if self.cluster_features == []:
            self.cluster_features.append(entry)
            return
        current_cf, index = self._find(entry, 'c')
        new_cf = current_cf + entry
        if new_cf.radius() < threshold:
            self.cluster_features[index] = new_cf
        else:
            self.cluster_features.append(entry)

    def seed_split(self, threshold):
        """Splits the node into two nodes.

        It choses the two clusters that are furthest apart, with regard to the chosen distance
        metric. These are the seeds, the two seeds are placed into seperate new nodes. The remaining
        entries are distributed between these nodes, based on the which cluster they are closest to.
        """
        seed_l = None
        seed_r = None
        max_dist = np.NINF
        for x, y in itr.permutations(self.cluster_features, r=2):
            if x.distance_metric(y) > max_dist:
                seed_l = x
                seed_r = y
        left = Node(self.order, [[seed_l], []])
        right = Node(self.order, [[seed_r], []])

        # when storing the max distance, also store the index
        # pop the entry from the list
        # then the first if statement can be removed
        # pop each element as we go?
        for i, cf in enumerate(self.cluster_features):
            if cf != seed_l and cf != seed_r:
                if cf.distance_metric(seed_l) < cf.distance_metric(seed_r):
                    left.add_entry(cf, threshold)
                else:
                    right.add_entry(cf, threshold)
        return left, right

    def _summarise(self):
        """
        Creates a summary Cluster Feature of the Cluster Features within the node.
        """
        return sum(cf for cf in self.cluster_features)

    def is_full(self):
        return len(self.cluster_features) == self.order

    def show(self, depth):
        print("Clusters at depth: {}".format(depth))
        for cf in self.cluster_features:
            cf.show()
        print("Children at depth: {}".format(depth))
        for child in self.children:
            child.show(depth + 1)
        # if self.children == []:
        #     return
        # print("Children:")
        # print
        # for child in self.children:
        #     pass


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
        left, right = node.seed_split(self.threshold)
        left_summary = left._summarise()
        right_summary = right._summarise()
        return [[left_summary, right_summary], [left, right]]

    def _merge(self, parent, index, cfs, children):
        # CFs within a node are not sorted as there is no condition to sort by
        # that would speed up insertions and retrievals

        parent.cluster_features[index] = cfs[0]
        parent.children[index] = children[0]

        print("inside merge")
        print(parent.cluster_features[:index] + \
            [cfs[1]] + parent.cluster_features[index:])
        print()

        parent.cluster_features = parent.cluster_features[:index] + \
            [cfs[1]] + parent.cluster_features[index:]
        parent.children = parent.children[:index] + \
            [children[1]] + parent.children[index:]

    def insert_point(self, X):
        """Inserts a point after traversing to a leaf node.

        If the leaf node is full, split the leaf node into two, and add to the parent node.
        Repeat up to the first ancestor with free space.

        Args:
            self (CFTree): A Cluster Feature tree instance.
            X (Vector): A vector/point that is to be added to a cluster in the tree. NOTE: This
            must be a numpy arrary.
        """

        parents = [[None, None]]
        child = self.root
        entry_cluster = ClusterFeature(n=1, ls=X, ss=np.sum(X**2))


        while child.children != []:
            parent = child
            child, index = child._find(entry_cluster, 'n')
            parents.append([parent, index])

        child.add_entry(entry_cluster, self.threshold)

        # index contains the position (index) of the child node in the parent
        # this must be modified to now represent one of the split nodes,
        # and another cf must be inserted into the parent to represent the another
        # split node

        # not sure this is sound syntax
        for parent, index in reversed(parents):
            if child.is_full():
                # we must split
                if parent is None:
                    # create new root
                    self.root = Node(
                        order=self.order, feature_children=self._split_summarise(child))
                    print("roots children after creation of new root")
                    print(self.root.children)
                    return
                else:
                    # we can insert the summaries into the parent (in the next iteration we check if that insertion made the parent full)
                    print("showing children before the split")
                    child.show(0)
                    split_sum = self._split_summarise(child)
                    print("parents children before merge with its children")
                    print(parent.children)
                    self._merge(parent, index, split_sum[0], split_sum[1])
                    print("parents children after merge with its children")
                    print(parent.children)
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

    def show(self):
        self.root.show(0)


def test_clusterfeature(points):
    cf = ClusterFeature(n=1, ls=points[0], ss=np.sum(points[0]**2))
    cf_2 = ClusterFeature(n=1, ls=points[1], ss=np.sum(points[1]**2))
    cf.show()
    cf_2.show()
    cf_3 = cf + cf_2
    cf_3.show()
    cf += cf_2
    cf.show()
    assert cf == cf_3
    cf_4 = ClusterFeature(n=1, ls=points[3], ss=np.sum(points[3]**2))
    cf_5 = ClusterFeature(n=1, ls=points[2], ss=np.sum(points[2]**2))
    cf_4.show()
    cf_5.show()
    print(cf_4.distance_metric(cf_5))
    assert cf_4.distance_metric(cf_5) == cf_5.distance_metric(cf_4)


def test_node(order, points, threshold):
    cf = ClusterFeature(n=1, ls=points[0], ss=np.sum(points[0]**2))
    cf_2 = ClusterFeature(n=1, ls=points[1], ss=np.sum(points[1]**2))
    n1 = Node(order)
    n1.add_entry(cf, threshold)
    n1.add_entry(cf_2, threshold)
    n1.show()
    cf_3 = ClusterFeature(n=1, ls=points[3], ss=np.sum(points[3]**2))
    cf_4 = ClusterFeature(n=1, ls=points[2], ss=np.sum(points[2]**2))
    n1.add_entry(cf_3, threshold)
    n1.add_entry(cf_4, threshold)
    n1.show()
    left, right = n1.seed_split(threshold)
    left.show()
    right.show()
    left_summary = left._summarise()
    print("Left summary:")
    left_summary.show()
    right_summary = right._summarise()
    print("Right summary:")
    right_summary.show()


def test_tree(order, points, threshold):
    tree = CFTree(order, threshold)
    for vector in points:
        tree.insert_point(vector)
    tree.show()
    tree.insert_point(np.array([11, 9]))
    tree.insert_point(np.array([10, 10]))
    tree.show()


def test_module():
    points = np.array([[2, 3], [2, 2], [1, 3], [10, 11], [11, 11], [10, 12]])
    threshold = 0.001
    order = 4
    # test_clusterfeature(points.copy())
    #test_node(order,points.copy(), threshold)
    test_tree(order, points, threshold)


if __name__ == '__main__':
    test_module()
