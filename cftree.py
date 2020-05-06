import numpy as np

class ClusterFeature(object):
    """Cluster Feature object.

    Stores the details of a sub-cluster.

    Attributes:
        n (int): The number of data points in the cluster.
        [ls] (float): The linear sum of the n data points.
        ss (float): The square sum of the n data points.
        [timestamps] (datetime): An arrary of size n_taxis, each element
        containing the timestamps at which this cluster was visited. The index represents the taxi id.
    """

    def __init__(self, n, ls, ss):
        self.n = n
        self.ls = ls
        self.ss = ss
        self.timestamps = []

    def centroid(self):
        return self.ls / self.n

    def radius(self):
        """
        Average distance from member points to the centroid.
        """
        xo = self.centroid()
        return ls**2 - 2 * ls * xo - xo**2


class Node(object):
    '''
    Base node object.

    Each node stores keys and values. Keys are not unique to each value, and
    as such values are stored as a list under each key.

    Attributes:
        order (int): Branching factor. The maximum number of keys each node can hold.
        values (ClusterFeature or Node): Contains CF(s) when the node is a leaf or pointers to the
        children in the case that it is not a leaf node.
    '''

    def __init__(self, order, threshold):
        self.order = order
        self.keys = []
        self.values = []
        self.threshold = threshold
        self.leaf = True

    def print_values(self):
        print(self.values)

    def add_point(self, X):
        """
        Adds the vector X to the closest CF according to a chosen distance
        metric (radius). Assumes that this is the correct node that the vector should be added
        to, i.e. with regard to the threshold.
        """
        cf = ClusterFeature(n=1, ls=sum(X), ss=sum(X**2))
        if not self.keys:
            self.keys.append(cf.radius())
            self.values.append(cf)
            return None

        (index,value) = (0,self.values[0])
        for i, item in enumerate(self.keys):
            # finds the closest leaf entry
            if item < value:
                index = i
                value = item

        # check if the closest CF can absorb 'cf' without violating the threshold
        absorbed_cf = ClusterFeature(value.n+cf.n,value.ls+cf.ls,value.ss+cf.ss)
        radius = absorbed_cf.radius()
        if radius < threshold:
            self.keys[index] = radius
            self.values[index] = absorbed_cf
            return
        # if it cannot be absorbed, the entry is added to the leaf
        if i+1 == len(self.keys):
            self.keys.append(cf.radius())
            self.values.append(cf)
            return

        self.keys = self.keys[:i] + [cf.radius()] + self.keys[i:]
        self.values = self.values[:i] + [cf] + self.values[i:]


    def add(self, key, value):
        '''
        Adds a key-value pair to the node.
        '''
        # if self.keys is empty
        if not self.keys:
            self.keys.append(key)
            self.values.append([value])
            return None

        for i, item in enumerate(self.keys):
            print("i: {}, item: {}".format(i, item))
            print("keys: {}, i: {}, self.keys[:i]: {}, self.keys[i:]: {}".format(
                self.keys, i, self.keys[:i], self.keys[i:]))
            if key == item:
                self.values[i].append(value)
                break

            elif key < item:
                self.keys = self.keys[:i] + [key] + self.keys[i:]
                self.values = self.values[:i] + [[value]] + self.values[i:]
                break

            elif i + 1 == len(self.keys):
                self.keys.append(key)
                self.values.append([value])
                break

    def mid_split(self):
        """
        Splits the node into two and stores them as children. Promotes the mid value
        to the parent node.
        """
        left = Node(self.order)
        right = Node(self.order)
        mid = self.order / 2

        left.keys = self.keys[:mid]
        left.values = self.values[:mid]

        right.keys = self.keys[mid+1:]
        right.values = self.values[mid+1:]
        
        self.keys = [self.keys[mid]]
        self.values = [left, right]
        self.leaf = False


    def split(self):
        '''
        Splits the node into two and stores them as child nodes.
        '''
        left = Node(self.order)
        right = Node(self.order)
        mid = self.order / 2

        left.keys = self.keys[:mid]
        left.values = self.values[:mid]

        right.keys = self.keys[mid:]
        right.values = self.values[mid:]

        print("-------")
        print(self.keys[mid:])
        print([right.keys[0]])
        print("-------")

        self.keys = [right.keys[0]]
        self.values = [left, right]
        self.leaf = False

    def is_full(self):
        '''
        Returns True if the node is full.
        '''
        return len(self.keys) == self.order

    def show(self, counter=0):
        '''
        Prints the keys at each level.
        '''
        print counter, str(self.keys)

        if not self.leaf:
            for item in self.values:
                item.show(counter + 1)


class BPlusTree(object):
    '''
    B+ tree object, consisting of nodes.

    Nodes will automatically be split into two once it is full. When a split
    occurs, a key will 'float' upwards and be inserted into the parent node to
    act as a pivot.

    Attributes:
        order (int): The maximum number of keys each node can hold.
    '''

    def __init__(self, order=8):
        self.root = Node(order)

    def _find(self, node, key):
        '''
        For a given node and key, returns the index where the key should be
        inserted and the list of values at that index.
        '''
        for i, item in enumerate(node.keys):
            if key < item:
                return node.values[i], i

        # node with i keys, has i+1 children
        return node.values[i + 1], i + 1

    def _merge(self, parent, child, index):
        '''
        For a parent and child node, extract a pivot from the child to be
        inserted into the keys of the parent. Insert the values from the child
        into the values of the parent.
        '''
        print("index: {}".format(index))
        parent.values.pop(index)
        pivot = child.keys[0]

        for i, item in enumerate(parent.keys):
            if pivot < item:
                parent.keys = parent.keys[:i] + [pivot] + parent.keys[i:]
                parent.values = parent.values[:i] + \
                    child.values + parent.values[i:]
                break

            elif i + 1 == len(parent.keys):
                parent.keys += [pivot]
                parent.values += child.values
                break

    def insert(self, key, value):
        '''
        Inserts a key-value pair after traversing to a leaf node. If the leaf
        node is full, split the leaf node into two.
        '''
        parent = None
        child = self.root

        while not child.leaf:
            parent = child
            child, index = self._find(child, key)

        child.add(key, value)

        if child.is_full():
            child.split()

            if parent and not parent.is_full():
                self._merge(parent, child, index)

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


def demo_node():
    print 'Initializing node...'
    node = Node(order=4)

    print '\nInserting key a...'
    node.add('a', 'alpha')
    print 'Is node full?', node.is_full()
    node.show()
    node.print_values()

    print '\nInserting keys b, c, d...'
    node.add('b', 'bravo')
    node.add('c', 'charlie')
    node.add('d', 'delta')
    print 'Is node full?', node.is_full()
    node.show()
    node.print_values()

    print '\nSplitting node...'
    node.split()
    node.show()


def demo_bplustree():
    print 'Initializing B+ tree...'
    bplustree = BPlusTree(order=3)

    print '\nB+ tree with 1 item...'
    bplustree.insert('a', 'alpha')
    bplustree.show()

    print '\nB+ tree with 2 items...'
    bplustree.insert('b', 'bravo')
    bplustree.show()

    print '\nB+ tree with 3 items...'
    bplustree.insert('c', 'charlie')
    bplustree.show()

    print '\nB+ tree with 4 items...'
    bplustree.insert('d', 'delta')
    bplustree.show()

    print '\nB+ tree with 5 items...'
    bplustree.insert('e', 'echo')
    bplustree.show()

    print '\nB+ tree with 6 items...'
    bplustree.insert('f', 'foxtrot')
    bplustree.show()

    print '\nRetrieving values with key e...'
    print bplustree.retrieve('e')


def test():
    tree = BPlusTree(order=2)
    tree.insert(4, 'four')
    tree.insert(5, 'five')
    tree.show()
    tree.insert(3, 'three')
    tree.insert(6, 'six')
    tree.insert(7, 'seven')
    tree.insert(8, 'eight')
    tree.insert(9, 'nine')
    tree.show()


if __name__ == '__main__':
    # demo_node()
    print '\n'
    # demo_bplustree()
    print '\n'
    test()
