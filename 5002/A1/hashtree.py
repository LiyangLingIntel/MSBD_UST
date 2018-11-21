# 
# -*- coding: utf-8 -*-
# Author： Leon Ling

from A1_llingab_20527456_Q1_code.settings import data_sets, max_leaf_size


def split(datasets, layer=0, max_size=max_leaf_size):

    # generate dict of hash tree recursively
    if len(datasets) > max_size:
        # length of coming dataset > max size, split it into dict
        # the key is result of number mod max size

        hash_tree = {}

        for item in datasets:
            index = item[layer] % max_size
            if index not in hash_tree:
                hash_tree[index] = []
            hash_tree[index].append(item)

        for key, subsets in hash_tree.items():
            hash_tree[key] = split(subsets, layer+1)

        return hash_tree

    else:   # length of coming dataset <= max size, return itself
        return datasets


def convertor(tree):
    """
    :param tree: hash_tree in dict
    :return: nested list
    """

    if isinstance(tree, list):
        return tree

    elif isinstance(tree, dict):
        list_tree = []
        # for item in tree.values():
        #     list_tree.append(convertor(item))

        # Specify the order of tree branches
        if tree.get(1):
            list_tree.append(convertor(tree[1]))
        if tree.get(2):
            list_tree.append(convertor(tree[2]))
        if tree.get(0):
            list_tree.append(convertor(tree[0]))

        return list_tree

    else:
        raise Exception('Tree must be in dict or list format!')


def generate_hash_tree(sets):
    
    tree_in_dict = split(sets)
    tree_in_list = convertor(tree_in_dict)

    return tree_in_list


if __name__ == "__main__":

    result = generate_hash_tree(data_sets)
    print(result)