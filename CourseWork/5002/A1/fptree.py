# 
# -*- coding: utf-8 -*-
# Author： Leon Ling

from A1_llingab_20527456_Q2_code.settings import support_threshold as threshold
from A1_llingab_20527456_Q2_code.settings import source_data
# import csv
# import json
# import pandas

class TreeNode(object):

    def __init__(self, item, parent=None, support=1):

        self.item = item
        self.parent = parent
        self.children = {}      # {'item_name': TreeNode, ...}  Using dict format to improve searching efficiency
        self.support = support

    def update_support(self):

        self.support += 1

    def get_child(self, child_name):
        """
        :param child_name:
        :return: if child exists return child TreeNode, else return None
        """
        return self.children.get(child_name, None)


class FPTree(object):

    def __init__(self, root, datasets, threshold, is_conditional=False):

        self.root = TreeNode(root)
        self.items_frequency = {}
        self.frequent_ordered_transactions = []
        self.header_table = {}

        self.data_pre_processing(datasets, threshold, is_conditional)
        self.generate_fp_tree()

    def data_pre_processing(self, source_data, threshold, is_conditional):
        """
        count frequent items, filter and reorder transactions in source data
        :param source_data: list of list of item names
        :param threshold: Int: minimum support
        :param is_conditional: Boolean: if true, add specific steps to handle the source data
        """
        data_sets = []
        # count frequency of each item
        for row in source_data:

            if is_conditional:
                counter = row[-1]       # row[-1] is support of this transaction
                row = row[:-2]          # row[-2] is the base item of conditional tree
            else:
                counter = 1

            transaction = []
            for item in row:
                if not item:    # item is empty string
                    break
                if item not in self.items_frequency:
                    self.items_frequency[item] = counter
                else:
                    self.items_frequency[item] += counter
                transaction.append(item)
            for i in range(counter):
                data_sets.append(transaction)
        del source_data

        for transaction in data_sets:
            candidate_trans = []
            for item in transaction:
                if self.items_frequency.get(item, 0) >= threshold:
                    candidate_trans.append(item)
            if candidate_trans:
                # sort transaction, frequency first, then alphabet
                self.frequent_ordered_transactions.append(sorted(candidate_trans, key=lambda x: (-self.items_frequency[x], x.lower())))

    def _add_node(self, parent_node, item_name):
        """
        For new_node that does not exist in current parent's children list
        Generate a new child node and add it to current parent node, return the new node
        :param parent_node: TreeNode
        :param item_name:  str: item name
        :return: TreeNode: new node
        """
        new_node = TreeNode(item_name, parent=parent_node)
        parent_node.children[item_name] = new_node

        if item_name not in self.header_table:
            self.header_table[item_name] = []
        self.header_table[item_name].append(new_node)

        return parent_node.children[item_name]

    def tree_grow(self, parent_node, transaction):
        """
        update fp_tree by coming transaction in recursive way
        :param transaction: list of str
        :param parent_node: TreeNode
        :return:
        """
        if len(transaction) < 1:
            return self
        else:
            current_item = transaction[0]
            item_node = parent_node.get_child(current_item)
            if item_node:   # item has existed below the parent node
                item_node.update_support()
                self.tree_grow(item_node, transaction[1:])
            else:   # item has not exist below the parent node
                item_node = self._add_node(parent_node, current_item)
                self.tree_grow(item_node, transaction[1:])

    def generate_fp_tree(self):

        for transaction in self.frequent_ordered_transactions:
            self.tree_grow(self.root, transaction)

        return self.root

    def has_single_path(self, node):
        """
        check current node has a single children path or not, return Boolean value
        """
        if len(node.children.keys()) == 0:
            return True
        elif len(node.children.keys()) == 1:
            return self.has_single_path(list(node.children.values())[0])
        else:
            return False

    def tree_traverser(self, node, to_root=True,):
        """
        traverse the fp-tree
        :param node: TreeNode, start node
        :param to_root: traverse to root or traverse to leaf
        :return: list of item names in str
        """
        result_list = []

        if to_root:     # return current item to root (without root)
            result_list.append(node.item)
            while node.parent.item:
                node = node.parent
                result_list.append(node.item)
        else:
            while node.children:        # return current node to leaf (without current node)
                result_list.append(list(node.children.values())[0].item)
                node = list(node.children.values())[0]

        return result_list

    def collect_cond_transactions(self, item_name):
        """
        return all transactions with support in fp-tree based on item_name
        :param item_name: str: base of conditional tree
        :return: list of list of item names and add the support at the tail
        """

        transaction_sets = []
        for item_node in self.header_table[item_name]:
            # generate list like [node, node, ..., node, support]
            transaction_sets.append(self.tree_traverser(item_node, to_root=True)[::-1] + [item_node.support])

        return transaction_sets


def generate_cond_fp_trees(fp_tree, min_support):

    cond_trees = {}

    for item_name in fp_tree.header_table.keys():
        transactions = fp_tree.collect_cond_transactions(item_name)
        cond_trees[item_name] = FPTree(root=None, datasets=transactions, threshold=min_support, is_conditional=True)

    return cond_trees


def generate_frequent_itemsets(cond_fp_tree, base, threshold):
    """
    generate frequent itemsets from given cond. fp-tree
    :param cond_fp_tree: FPTree:
    :param base: str: item name
    :param threshold: Int: minimum support
    :return: list of item names
    """
    def generate_frequent_sub_patterns(sub_item_set):
        """
        generate all sub-sets of set: 'sub_item_set'
        """
        frequent_set = [[]]
        for item in sub_item_set:
            frequent_set.extend([subset+[item] for subset in frequent_set])
        return frequent_set

    # generate frequent patterns recursively
    if cond_fp_tree.has_single_path(cond_fp_tree.root):     # if cond. tree is single path, generate fp sets
        sub_items = cond_fp_tree.tree_traverser(cond_fp_tree.root, to_root=False)
        return list(map(lambda x: [base]+x, generate_frequent_sub_patterns(sub_items)))
    else:       # if not, let base union all subsets of the union of result of its cond. fp-trees
        cond_sub_fp_trees = generate_cond_fp_trees(cond_fp_tree, threshold)
        sub_set_container = []
        for sub_base, sub_tree in cond_sub_fp_trees.items():
            sub_set_container.extend(generate_frequent_itemsets(sub_tree, sub_base, threshold))
        sub_set_container = list(map(lambda x: [base] + x, [[]]+sub_set_container))
        return sub_set_container


def gen_all_fp_itemsets(cond_trees, min_support):

    fp_itemsets = []
    for item, cond_tree in cond_trees.items():
        fp_itemsets.extend(generate_frequent_itemsets(cond_tree, item, min_support))

    return fp_itemsets


def output_frequent_items(fps):
    """
    Output frequent itemsets to .csv file as Q2(a) answer
    :param fp: list of list of frequent item in str
    """
    with open('./frequent_groceries.csv', 'w') as file:
        for fp_set in fps:
            # other csv writer of csv lib or pandas lib cannot support the required format in Sample_submission.csv
            # "{'item_name', 'item_name')", some ' or " will be removed by some csv characters whatever the input is
            # or strings. Common file.write works well.
            file.write('"' + str(set(fp_set)) + '"\n')


def output_conditional_fp_trees(cond_trees, show_base=False):
    """
    print conditional fp-trees in required format
    :param cond_trees: dict: key is conditional base name, value is fp-tree object
    :param show_base: to print conditional base or not
    """
    def interpret_fp_tree(node):
        tree_in_list = []
        root_item = 'Null Set' if node.item is None else node.item
        tree_in_list.append(root_item+' '+str(node.support))
        if node.children:
            child_list = []
            for child in node.children.values():
                child_list.append(interpret_fp_tree(child))
            tree_in_list.append(child_list)
            return tree_in_list
        else:
            if node.item is None:
                return tree_in_list
            return node.item + ' ' + str(node.support)

    for fp_base, fp_tree in cond_trees.items():
        tree_in_list = interpret_fp_tree(fp_tree.root)
        if len(tree_in_list) > 1:
            if show_base:
                print(f'{fp_base}: {interpret_fp_tree(fp_tree.root)}')
            else:
                print(interpret_fp_tree(fp_tree.root))


if __name__ == '__main__':

    # root = TreeNode('root')
    fp_tree = FPTree(root=None, datasets=source_data, threshold=threshold)

    conditional_fp_trees = generate_cond_fp_trees(fp_tree, threshold)
    fp_itemsets = gen_all_fp_itemsets(conditional_fp_trees, threshold)

    output_frequent_items(fp_itemsets)
    output_conditional_fp_trees(conditional_fp_trees)
