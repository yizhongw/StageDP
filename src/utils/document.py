#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午9:06
from os.path import isfile

from utils.token import Token


class Doc(object):
    """ Build one doc instance from *.merge file
    """

    def __init__(self):
        """
        """
        self.token_dict = None
        self.edu_dict = None
        self.rel_paris = None
        self.fmerge = None

    def read_from_fmerge(self, fmerge):
        """ Read information from the merge file, and create an Doc instance
        :type fmerge: string
        :param fmerge: merge file name
        """
        self.fmerge = fmerge
        if not isfile(fmerge):
            raise IOError("File doesn't exist: {}".format(fmerge))
        gidx, self.token_dict = 0, {}
        with open(fmerge, 'r') as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                tok = self._parse_fmerge_line(line)
                self.token_dict[gidx] = tok
                gidx += 1
        # Get EDUs from tokendict
        self.edu_dict = self._recover_edus(self.token_dict)

    def init_from_tokens(self, token_list):
        self.token_dict = {idx: token for idx, token in enumerate(token_list)}
        self.edu_dict = self._recover_edus(self.token_dict)

    @staticmethod
    def _parse_fmerge_line(line):
        """ Parse one line from *.merge file
        """
        items = line.split("\t")
        tok = Token()
        tok.pidx, tok.sidx, tok.tidx = int(items[-1]), int(items[0]), int(items[1])
        # Without changing the case
        tok.word, tok.lemma = items[2], items[3]
        tok.pos = items[4]
        tok.dep_label = items[5]
        try:
            tok.hidx = int(items[6])
        except ValueError:
            pass
        tok.ner, tok.partial_parse = items[7], items[8]
        try:
            tok.eduidx = int(items[9])
        except ValueError:
            print("EDU index for {} is missing in fmerge file".format(tok.word))
            # sys.exit()
            pass
        return tok

    def to_conll(self):
        conll_str = ''
        for idx, token in self.token_dict.items():
            conll_str += '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(token.sidx, token.tidx, token.word,
                                                                               token.lemma, token.pos,
                                                                               token.dep_label, token.hidx, token.ner,
                                                                               token.partial_parse, token.eduidx,
                                                                               token.pidx)
        return conll_str

    @staticmethod
    def write_line(token_list, file):
        with open(file, 'w') as fout:
            for token in token_list:
                fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(token.sidx, token.tidx, token.word,
                                                                                 token.lemma, token.pos,
                                                                                 token.dep_label, token.hidx, token.ner,
                                                                                 token.partial_parse, token.eduidx,
                                                                                 token.pidx))

    @staticmethod
    def _recover_edus(token_dict):
        """ Recover EDUs from token_dict
        """
        N, edu_dict = len(token_dict), {}
        for gidx in range(N):
            token = token_dict[gidx]
            eidx = token.eduidx
            try:
                val = edu_dict[eidx]
                edu_dict[eidx].append(gidx)
            except KeyError:
                edu_dict[eidx] = [gidx]
        return edu_dict


if __name__ == '__main__':
    doc = Doc()
    fmerge = "../data/training/file1.merge"
    doc.read_from_fmerge(fmerge)
    print(len(doc.edudict))
