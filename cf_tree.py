import numpy as np

# NOTE: Branching factor cannot be 1 and possibly not 2 either, for performance reasons?
# TODO: Define + operation for the Cluster Feature: https://realpython.com/operator-function-overloading/

class Node(object):
    """docstring for Node."""

    def __init__(self, order=8, feature_children=None):
        self.order = order
        if feature_children is None:
            self.cluster_features = []
            self.children = []
        else:
            self.cluster_features = feature_children[0]
            self.children = feature_children[1]

    # could be merged with find node function inside CFTree
    def _find_cluster(self, entry):
        (index, distance) = (-1, np.inf)
        for i, item in enumerate(self.cluster_features):
            centroid_distance = item.distance_metric(entry)
            if centroid_distance < distance:
                index = i
                distance = centroid_distance
        return self.cluster_features[index], index

    def add_entry(self, entry, threshold):
        current_cf, index = self._find_cluster(entry)
        new_cf = ClusterFeature(current_cf+entry)
        if new_cf.diameter() < threshold:
            self.cluster_features[index] = new_cf
        else:
            self.cluster_features.append(entry)

class CFTree(object):
    """
    Clustering Feature tree object, consisting of nodes.

    Nodes will automatically be split into two when they are full. If the root
    is split, the height increases by one.

    When a split occurs, summaries (Cluster Features) of all sub Cluster Features within each
    node will be created and subsequently inserted into the parent node. If the
    parent is full we must split it. Then is applied recursively until we find space
    for the node. i.e. If the grand-parent node is also full.

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

    # this function and its split derivative might be best suited in the Node class
    def _summarise(self, node):
        """
        Creates a summary Cluster Feature of the Cluster Features within a given node.
        """
        # (n, ls, ss) = sum( (cf.n, cf.ls, cf.ss) for cf in node.cluster_features)
        n, ls, ss = (0, 0, 0)
        for cf in node.cluster_features:
            n += cf.n
            ls += cf.ls
            ss += cf.ss
        return ClusterFeature(n, ls, ss)

    def _split_summarise(self, node):
        left, right = node.seed_split()
        # creates two cluster features summarising the left and right child
        left_summary = self._summarise(left)
        right_summary = self._summarise(right)
        #left_summary = ClusterFeature(left.cluster_features)
        #right_summary = ClusterFeature(right.cluster_features)
        return ((left_summary, right_summary), (left, right))

    def _find_node(self, node, entry_cluster):
        (index, cluster, distance) = (-1, None, np.inf)
        for i, item in enumerate(node.cluster_features):
            centroid_distance = item.distance_metric(entry_cluster)
            if centroid_distance < distance:
                index = i
                cluster = item
                distance = centroid_distance
        return node.children[index], index

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
        """
        Inserts a point after traversing to a leaf node. If the leaf
        node is full, split the leaf node into two.
        """

        parents = [(None, None)]
        child = self.root
        entry_cluster = ClusterFeature(n=1, ls=sum(X), ss=sum(X**2))

        while child.children != []:
            parent = child
            child, index = self._find_node(child, entry_cluster)
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
                    self.root = Node(order=self.order, self._split_summarise(child))
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
                    parent.cluster_features[index] = self._summarise(child)
                    child = parent

        # refinement step done here
