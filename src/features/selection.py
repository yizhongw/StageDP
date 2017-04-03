#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 上午11:26
from operator import itemgetter

import numpy
from scipy.stats import entropy


class FeatureSelector(object):
    """ Feature selection module
    """

    def __init__(self, topn, thresh, method='frequency'):
        """ Initialization
        """
        self.method = method
        self.topn = topn
        self.thresh = thresh

    def select(self, features, freq_table):
        """ Select features via some criteria

        :type features: dict
        :param features: features vocab

        :type freq_table: 2-D numpy.array
        :param freq_table: frequency table with rows as features,
                          columns as frequency values
        """
        if self.method == 'frequency':
            feat_vals = self.frequency(features, freq_table)
        elif self.method == 'entropy':
            feat_vals = self.entropy(features, freq_table)
        elif self.method == 'freq-entropy':
            feat_vals = self.freq_entropy(features, freq_table)
        else:
            raise KeyError("Unrecognized method")
        new_features = self.rank(feat_vals)
        return new_features

    def rank(self, feat_vals):
        """ Rank all features and take top-n features

        :type feat_vals: dict
        :param feat_vals: {features:value}
        """
        features = {}
        sorted_vals = sorted(feat_vals.items(), key=itemgetter(1))
        sorted_vals = sorted_vals[::-1]
        for (idx, item) in enumerate(sorted_vals):
            if 0 < self.topn <= idx or item[1] < self.thresh:
                break
            features[item[0]] = idx
        return features

    def frequency(self, features, freq_table):
        """ Compute frequency values of features
        """
        feat_vals = {}
        for (feat, idx) in features.items():
            feat_vals[feat] = freq_table[idx, :].sum()
        return feat_vals

    def entropy(self, features, freq_table):
        """
        """
        feat_vals = {}
        for (feat, idx) in features.items():
            freq = freq_table[idx, :]
            feat_vals[feat] = 1 / (entropy(freq) + 1e-3)
        return feat_vals

    def freq_entropy(self, features, freq_table):
        """
        """
        feat_vals = {}
        feat_freqs = self.frequency(features, freq_table)
        feat_ents = self.entropy(features, freq_table)
        for feat in features.keys():
            freq = feat_freqs[feat]
            ent = feat_ents[feat]
            feat_vals[feat] = numpy.log(freq + 1e-3) * (ent + 1e-3)
        return feat_vals


def test():
    vocab = {'hello': 0, 'data': 1, 'computer': 2}
    freq_table = [[23, 23, 23, 23], [23, 1, 4, 5], [1, 34, 1, 1]]
    freq_table = numpy.array(freq_table)
    fs = FeatureSelector(topn=2, method='freq-entropy')
    newvocab = fs.select(vocab, freq_table)
    print(newvocab)


if __name__ == '__main__':
    test()
