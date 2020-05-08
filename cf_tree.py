import numpy as np


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

    def _split_summarise(node):
        left, right = node.seed_split()
        # creates two cluster features summarising the left and right child
        left_summary = ClusterFeature(left.cluster_features)
        right_summary = ClusterFeature(right.cluster_features)
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
        """
        Inserts a point after traversing to a leaf node. If the leaf
        node is full, split the leaf node into two.
        """
        parent = None
        child = self.root

        entry_cluster = ClusterFeature(n=1, ls=sum(X), ss=sum(X**2))

        while child.children != []:
            parent = child
            child, index = self._find_node(child, entry_cluster)

        child.add_entry(entry_cluster)

        # index contains the position (index) of the child node in the parent
        # this must be modified to now represent one of the split nodes,
        # and another cf must be inserted into the parent to represent the another
        # split node

        if child.is_full():
            if parent is None:
                self.root = Node(order=self.order, _split_summarise(child))
                return
            # traverse up the tree
            for parent in parents:
                # assert(child.is_full())
                if parent.is_full() is False:
                    # we can safely insert the summaries into the parent
                    self._merge(parent, index, _split_summarise(child))
                    return
                else:
                    if parent is None:
                        # we have hit the root
                        # assert(child==self.root)
                        self.root = Node(order=self.order, _split_summarise(child))
                        return
                    else:
                        # we can insert the summaries into the parent
                        self._merge(parent, index, _split_summarise(child))
                        child = parent


        for parent in parents:
            if child.is_full():
                # we must split
                if parent is None:
                    # create new root
                    self.root = Node(order=self.order,_split_summarise(child))
                    return
                else:
                    # we can insert the summaries into the parent
                    self._merge(parent, index, _split_summarise(child))
                    child = parent
            else:
                # no spliting, but updating the path from leaf to root
                if parent is None:
                    return
                else:
                    self._merge(parent,index,_summarise(child))
                    child = parent

        # refinement step done here

    def show(self):
        """
        Prints the keys at each level.
        """
        self.root.show()
