#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午8:45
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize


class ParseError(Exception):
    """ Exception for parsing
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class ActionError(Exception):
    """ Exception for illegal parsing action
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def reverse_dict(dct):
    """ Reverse the {key:val} in dct to
        {val:key}
    """
    newmap = {}
    for (key, val) in dct.items():
        newmap[val] = key
    return newmap


def str2action(action_str):
    """ Transform label to action
    """
    items = action_str.split('-')
    if len(items) == 1:
        action = (items[0], None, None)
    elif len(items) == 3:
        action = tuple(items)
    elif len(items) > 3:
        relalabel = '-'.join(items[2:])
        action = tuple((items[0], items[1], relalabel))
    else:
        raise ValueError("Unrecognized label: {}".format(action_str))
    return action


def action2str(action):
    """ Transform action into action_str
    """
    if action[0] == 'Shift':
        action_str = action[0]
    elif action[0] == 'Reduce':
        action_str = '-'.join(list(action))
    else:
        raise ValueError("Unrecognized parsing action: {}".format(action))
    return action_str


def vectorize(features, vocab):
    """ Transform a features list into a numeric vector
        with a given vocab

    :type dpvocab: dict
    :param dpvocab: vocab for distributional representation

    :type projmat: scipy.lil_matrix
    :param projmat: projection matrix for disrep
    """
    vec = lil_matrix((1, len(vocab)))

    for feat in features:
        try:
            fidx = vocab[feat]
            vec[0, fidx] += 1.0
        except KeyError:
            pass
    # Normalization
    vec = normalize(vec)
    return vec


class2rel = {
    'Attribution': ['attribution', 'attribution-e', 'attribution-n', 'attribution-negative'],
    'Background': ['background', 'background-e', 'circumstance', 'circumstance-e'],
    'Cause': ['cause', 'cause-result', 'result', 'result-e', 'consequence', 'consequence-n-e', 'consequence-n',
              'consequence-s-e', 'consequence-s'],
    'Comparison': ['comparison', 'comparison-e', 'preference', 'preference-e', 'analogy', 'analogy-e', 'proportion'],
    'Condition': ['condition', 'condition-e', 'hypothetical', 'contingency', 'otherwise'],
    'Contrast': ['contrast', 'concession', 'concession-e', 'antithesis', 'antithesis-e'],
    'Elaboration': ['elaboration-additional', 'elaboration-additional-e', 'elaboration-general-specific-e',
                    'elaboration-general-specific', 'elaboration-part-whole', 'elaboration-part-whole-e',
                    'elaboration-process-step', 'elaboration-process-step-e', 'elaboration-object-attribute-e',
                    'elaboration-object-attribute', 'elaboration-set-member', 'elaboration-set-member-e', 'example',
                    'example-e', 'definition', 'definition-e'],
    'Enablement': ['purpose', 'purpose-e', 'enablement', 'enablement-e'],
    'Evaluation': ['evaluation', 'evaluation-n', 'evaluation-s-e', 'evaluation-s', 'interpretation-n',
                   'interpretation-s-e', 'interpretation-s', 'interpretation', 'conclusion', 'comment', 'comment-e',
                   'comment-topic'],
    'Explanation': ['evidence', 'evidence-e', 'explanation-argumentative', 'explanation-argumentative-e', 'reason',
                    'reason-e'],
    'Joint': ['list', 'disjunction'],
    'Manner-Means': ['manner', 'manner-e', 'means', 'means-e'],
    'Topic-Comment': ['problem-solution', 'problem-solution-n', 'problem-solution-s', 'question-answer',
                      'question-answer-n', 'question-answer-s', 'statement-response', 'statement-response-n',
                      'statement-response-s', 'topic-comment', 'comment-topic', 'rhetorical-question'],
    'Summary': ['summary', 'summary-n', 'summary-s', 'restatement', 'restatement-e'],
    'Temporal': ['temporal-before', 'temporal-before-e', 'temporal-after', 'temporal-after-e', 'temporal-same-time',
                 'temporal-same-time-e', 'sequence', 'inverted-sequence'],
    'Topic-Change': ['topic-shift', 'topic-drift'],
    'Textual-Organization': ['textualorganization'],
    'span': ['span'],
    'Same-Unit': ['same-unit']
}

rel2class = {}
for cl, rels in class2rel.items():
    rel2class[cl.lower()] = cl
    for rel in rels:
        rel2class[rel.lower()] = cl
