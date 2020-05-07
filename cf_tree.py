

class CFTree(object):
    '''
    Clustering Feature tree object, consisting of nodes.

    Nodes will automatically be split into two once it is full. When a split
    occurs, a key will 'float' upwards and be inserted into the parent node to
    act as a pivot.

    Attributes:
        order (int): The maximum number of keys each node can hold.
        threshold (float): The radius of each cluster must be less than this value.
    '''

    def __init__(self, order=8, threshold):
        self.root = Node(order)
        self.threshold = threshold

    def _find_node(self, node, entry_cluster):
        # must check the keys because values are not always the same type
        # depends on if the node is a leaf or not
        (index, cluster, distance) = (-1, None, np.inf)
        for i, item in enumerate(node.keys):
            centroid_distance = entry_cluster.d0_float(item)
            if centroid_distance < distance:
                # can immediately return the cluster?
                index = i
                cluster = item
                distance = centroid_distance
        return node.values[index], index

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

        # TODO: Refactor add_point to take in a cluster feature, or create two
        #       methods one that takes in a position, another that takes a
        #       cluster.
        #child.add_point(X)

        child.add_entry(entry_cluster)

        # index contains the position (index) of the child node in the parent
        # this must be modified to now represent one of the split nodes,
        # and another cf must be inserted into the parent to represent the another
        # split node

        if child.is_full():
            if parent is None:
                left, right = child.seed_split()
                # creates a cluster feature summarising the left and right child
                left_summary = ClusterFeature(left.cluster_features)
                right_summary = ClusterFeature(right.cluster_features)
                self.root = Node(order=self.order,[left_summary,right_summary],[left,right])
                return
            # traverse up the tree
            for parent in parents:
                # assert(child.is_full())
                if parent.is_full() is False:
                    left, right = child.seed_split()
                    # creates a cluster feature summarising the left and right child
                    left_summary = ClusterFeature(left.cluster_features)
                    right_summary = ClusterFeature(right.cluster_features)
                    # we can safely insert the summaries into the parent
                    self._merge(parent,index,[left_summary,right_summary],[left,right])
                    break
                else:
                    left, right = child.seed_split()
                    # creates a cluster feature summarising the left and right child
                    left_summary = ClusterFeature(left.cluster_features)
                    right_summary = ClusterFeature(right.cluster_features)
                    # we have hit the root
                    if parent is None:
                        # assert(child==self.root)
                        self.root = Node(order=self.order,[left_summary,right_summary],[left,right])
                        break
                    else:
                        # we can safely insert the summaries into the parent
                        self._merge(parent,index,[left_summary,right_summary],[left,right])
                        child = parent

        # refinement step done here

    def retrieve(self, key):
        '''
        Returns a value for a given key, and None if the key does not exist.
        '''
        child = self.root

        while not child.leaf:
            child, index = self._find(child, key)

        for i, item in enumerate(child.keys):
            if key == item:
                return child.values[i]

        return None

    def show(self):
        '''
        Prints the keys at each level.
        '''
        self.root.show()
