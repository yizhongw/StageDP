#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 下午8:04
import os

from eval.metrics import Metrics
from models.parser import RstParser
from models.tree import RstTree
from utils.document import Doc


class Evaluator(object):
    def __init__(self, model_dir='../data/model'):
        print('Load parsing models ...')
        self.parser = RstParser()
        self.parser.load(model_dir)

    def parse(self, doc):
        """ Parse one document using the given parsing models"""
        pred_rst = self.parser.sr_parse(doc)
        return pred_rst

    @staticmethod
    def writebrackets(fname, brackets):
        """ Write the bracketing results into file"""
        # print('Writing parsing results into file: {}'.format(fname))
        with open(fname, 'w') as fout:
            for item in brackets:
                fout.write(str(item) + '\n')

    def eval_parser(self, path='./examples', report=False, bcvocab=None, draw=True):
        """ Test the parsing performance"""
        # Evaluation
        met = Metrics(levels=['span', 'nuclearity', 'relation'])
        # ----------------------------------------
        # Read all files from the given path
        doclist = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.merge')]
        pred_forms = []
        gold_forms = []
        depth_per_relation = {}
        for fmerge in doclist:
            # ----------------------------------------
            # Read *.merge file
            doc = Doc()
            doc.read_from_fmerge(fmerge)
            # ----------------------------------------
            # Parsing
            pred_rst = self.parser.sr_parse(doc, bcvocab)
            if draw:
                pred_rst.draw_rst(fmerge.replace(".merge", ".ps"))
            # Get brackets from parsing results
            pred_brackets = pred_rst.bracketing()
            fbrackets = fmerge.replace('.merge', '.brackets')
            # Write brackets into file
            Evaluator.writebrackets(fbrackets, pred_brackets)
            # ----------------------------------------
            # Evaluate with gold RST tree
            if report:
                fdis = fmerge.replace('.merge', '.dis')
                gold_rst = RstTree(fdis, fmerge)
                gold_rst.build()
                met.eval(gold_rst, pred_rst)
                for node in pred_rst.postorder_DFT(pred_rst.tree, []):
                    pred_forms.append(node.form)
                for node in gold_rst.postorder_DFT(gold_rst.tree, []):
                    gold_forms.append(node.form)

                nodes = gold_rst.postorder_DFT(gold_rst.tree, [])
                inner_nodes = [node for node in nodes if node.lnode is not None and node.rnode is not None]
                for idx, node in enumerate(inner_nodes):
                    relation = node.rnode.relation if node.form == 'NS' else node.lnode.relation
                    rela_class = RstTree.extract_relation(relation)
                    if rela_class in depth_per_relation:
                        depth_per_relation[rela_class].append(node.depth)
                    else:
                        depth_per_relation[rela_class] = [node.depth]
                    lnode_text = ' '.join([gold_rst.doc.token_dict[tid].word for tid in node.lnode.text])
                    lnode_lemmas = ' '.join([gold_rst.doc.token_dict[tid].lemma for tid in node.lnode.text])
                    rnode_text = ' '.join([gold_rst.doc.token_dict[tid].word for tid in node.rnode.text])
                    rnode_lemmas = ' '.join([gold_rst.doc.token_dict[tid].lemma for tid in node.rnode.text])
                    # if rela_class == 'Topic-Change':
                    #     print(fmerge)
                    #     print(relation)
                    #     print(lnode_text)
                    #     print(rnode_text)
                    #     print()

        if report:
            met.report()
            # print(Counter(pred_forms))
            # print(Counter(gold_forms))
            # for relation, depths in depth_per_relation.items():
            #     print('{} {}'.format(relation, sum(depths) / len(depths)))
