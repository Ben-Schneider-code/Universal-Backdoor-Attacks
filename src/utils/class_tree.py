import numpy as np
from matplotlib import pyplot as plt


class Node:
    def __init__(self, tree, data, depth, feature_order):
        self.data = data
        self.depth = depth
        self.vector_index = feature_order[depth]
        self.left=None
        self.right=None
        self.split_value = tree.feature_means[self.vector_index]
        self.tree = tree
        if depth < tree.max_depth:

            left_data = []
            right_data = []

            for class_iterator in data:

                if(class_iterator[1][self.vector_index] < self.split_value):
                    left_data.append(class_iterator)
                else:
                    right_data.append(class_iterator)
            self.left = Node(tree, left_data, depth+1,feature_order)
            self.right = Node(tree, right_data, depth+1,feature_order)

    def __str__(self):
        tree_str = ""
        if self.left is not None:
            tree_str = tree_str +  str(self.left)
        if self.right is not None:
            tree_str = tree_str + str(self.right)

        return "\nDepth: " + str(self.depth) + " split value: " + str(self.split_value) +  " Size: " + str(len(self.data)) + tree_str

    def leaf(self):
        tree_str = ""
        if self.left is not None:
            tree_str = tree_str + (self.left.leaf())
        if self.right is not None:
            tree_str = tree_str + str(self.right.leaf())

        if self.depth == self.tree.max_depth:
            print( "\nDepth: " + str(self.depth) + " split value: " + str(self.split_value) + " Size: " + str(len(self.data)) + tree_str)

        return tree_str

    def leaf_size(self, size_list, depth):

        if self.left is not None:
            self.left.leaf_size(size_list,depth)
        if self.right is not None:
            self.right.leaf_size(size_list,depth)

        if self.depth == depth:
            size_list.append(len(self.data))

        return size_list


class ClassTree:

    def __init__(self, class_means, max_depth, feature_order):
            self.class_means = class_means
            self.max_depth = max_depth
            class_stack = np.stack([item[1] for item in class_means])
            print(class_stack.shape)
            self.feature_means = np.median(class_stack, axis=0)
            print(self.feature_means.shape)
            self.root = Node(self, class_means, 0, feature_order)

    def __str__(self):
        return str(self.root)


    def leaf_str(self):
        return self.root.leaf()

    def leaf_size_list(self, depth=None):

        if depth is None:
            depth = self.max_depth

        return self.root.leaf_size([],depth)

    def hist(self):
        data = self.leaf_size_list()
        print(sum(data)/len(data))
        plt.hist(data,bins=100, edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Most common leaf sizes')
        plt.show()
