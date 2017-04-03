#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 17-1-17 下午8:52
from collections import Counter

from data_helper import DataHelper
from models.tree import RstTree
from utils.other import class2rel


def cal_class_distribution(data_dir, level):
    """
    calculate the class distribution
    :param data_dir:
    :param level: 0 for inner-sentence, 1 for inter-sentence but inner paragraph, 2 for inter-paragraph, 3 for different depth
    :return: None
    """
    rst_trees = DataHelper.read_rst_trees(data_dir)
    all_nodes = [node for rst_tree in rst_trees for node in rst_tree.postorder_DFT(rst_tree.tree, [])]
    if level in [0, 1, 2]:
        valid_relations = [RstTree.extract_relation(node.child_relation) for node in all_nodes if
                           node.level == level and node.child_relation is not None]
        distribution = Counter(valid_relations)
        for cla in class2rel:
            if cla not in distribution:
                distribution[cla] = 0
        return distribution
    if level == 3:
        depth_relation_distributions = {}
        for node in all_nodes:
            if node.lnode is None and node.rnode is None:
                continue
            if node.depth in depth_relation_distributions:
                depth_relation_distributions[node.depth][RstTree.extract_relation(node.child_relation)] += 1
            else:
                depth_relation_distributions[node.depth] = Counter()
                depth_relation_distributions[node.depth][RstTree.extract_relation(node.child_relation)] = 1
        for depth, distribution in depth_relation_distributions.items():
            for cla in class2rel:
                if cla not in distribution:
                    distribution[cla] = 0
        return depth_relation_distributions


if __name__ == '__main__':
    data_dir = '../../data/rst_discourse_treebank/data/RSTtrees-WSJ-main-1.0/TRAINING/'
    for level in [0, 1, 2, 3]:
        if level == 3:
            depth_relation_distributions = cal_class_distribution(data_dir, level)
            print('Distribution for level {}'.format(level))
            for depth, distribution in depth_relation_distributions.items():
                print('Distribution for depth {}'.format(depth))
                for relation, cnt in distribution.items():
                    print('{}\t{}'.format(relation, cnt))
                print()
        else:
            distribution = cal_class_distribution(data_dir, level)
            print('Distribution for level {}:'.format(level))
            for relation, cnt in distribution.items():
                print('{}\t{}'.format(relation, cnt))
            print()
