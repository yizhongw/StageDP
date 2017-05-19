#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: yizhong
# created_at: 17-5-19 下午7:13
import os
import gzip
import pickle
import argparse
from pycorenlp import StanfordCoreNLP
from models.parser import RstParser
from utils.token import Token
from utils.document import Doc
from nltk import Tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edu_file_dir')
    parser.add_argument('--output_dir')
    return parser.parse_args()


def create_doc_from_edu_file(edu_file, annotate_func):
    with open(edu_file, 'r') as fin:
        doc_tokens = []
        paragraphs = [p.strip() for p in fin.read().split('<P>') if p.strip()]
        previous_edu_num = 0
        for pidx, para in enumerate(paragraphs):
            sentences = [s.strip() for s in para.split('<S>') if s.strip()]
            for sidx, sent in enumerate(sentences):
                edus = [e.strip() + ' ' for e in sent.split('\n') if e.strip()]
                sent_text = ''.join(edus)
                annot_re = annotate_func(sent_text)['sentences'][0]
                sent_tokens = []
                for t in annot_re['tokens']:
                    token = Token()
                    token.tidx, token.word, token.lemma, token.pos = t['index'], t['word'], t['lemma'], t['pos']
                    token.pidx, token.sidx = pidx + 1, sidx
                    edu_text_length = 0
                    for eidx, edu_text in enumerate(edus):
                        edu_text_length += len(edu_text)
                        if edu_text_length > t['characterOffsetEnd']:
                            token.eduidx = previous_edu_num + eidx + 1
                            break
                    sent_tokens.append(token)
                for dep in annot_re['basicDependencies']:
                    dependent_token = sent_tokens[dep['dependent']-1]
                    dependent_token.hidx = dep['governor']
                    dependent_token.dep_label = dep['dep']
                doc_tokens += sent_tokens
                previous_edu_num += len(edus)
    doc = Doc()
    doc.init_from_tokens(doc_tokens)
    return doc


def main():
    args = parse_args()
    parser = RstParser()
    parser.load('../data/model')
    with gzip.open('../data/resources/bc3200.pickle.gz') as fin:
        print('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    core_nlp = StanfordCoreNLP('http://localhost:9000')
    annotate = lambda x: core_nlp.annotate(x, properties={
        'annotators': 'tokenize,ssplit,pos,lemma,parse,depparse',
        'outputFormat': 'json',
        'ssplit.isOneSentence': True
    })
    edu_file_list = [os.path.join(args.edu_file_dir, fname) for fname in os.listdir(args.edu_file_dir) if fname.endswith('.edu.txt')]
    for edu_file in edu_file_list:
        print('Parsing {}...'.format(edu_file))
        doc = create_doc_from_edu_file(edu_file, annotate_func=annotate)
        pred_rst = parser.sr_parse(doc, brown_clusters)
        tree_str = pred_rst.get_parse()
        pprint_tree_str = Tree.fromstring(tree_str).pformat(margin=150)
        with open(os.path.join(args.output_dir, os.path.basename(edu_file) + '.parse'), 'w') as fout:
            fout.write(pprint_tree_str)


if __name__ == '__main__':
    main()