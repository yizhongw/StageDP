#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 上午10:32
from utils.other import action2str


class ActionFeatureGenerator(object):
    def __init__(self, stack, queue, action_hist, doc, bcvocab, nprefix=11):
        """ Initialization of features generator

        :type stack: list
        :param stack: list of Node instance

        :type queue: list
        :param queue: list of Node instance

        :type doc: Doc instance
        :param doc:
        """
        # Predefined variables
        self.npref = nprefix
        # Load Brown clusters
        self.bcvocab = bcvocab
        # -------------------------------------
        self.action_hist = action_hist
        self.doc = doc
        self.stack = stack
        self.queue = queue
        # Stack
        if len(stack) >= 2:
            self.top1span, self.top2span = stack[-1], stack[-2]
        elif len(stack) == 1:
            self.top1span, self.top2span = stack[-1], None
        else:
            self.top1span, self.top2span = None, None
        # Queue
        if len(queue) > 0:
            self.firstspan = queue[0]
        else:
            self.firstspan = None
        # Doc length wrt EDUs
        self.doclen = len(self.doc.edu_dict)

    def gen_features(self):
        """ Main function to generate features
        """
        feat_list = []
        # Status features (Basic features)
        for feat in self.status_features():
            feat_list.append(feat)
        # Operational features
        for feat in self.operational_features():
            feat_list.append(feat)
        # Textual organization features
        for feat in self.organizational_features():
            feat_list.append(feat)
        # Syntactic features
        for feat in self.syntactic_featues():
            feat_list.append(feat)
        # length features
        for feat in self.structural_features():
            feat_list.append(feat)
        # Lexical features
        for feat in self.ngram_features():
            feat_list.append(feat)
        # Nucleus features
        for feat in self.nucleus_features():
            feat_list.append(feat)
        # Brown clusters
        if self.bcvocab is not None:
            for feat in self.bc_features():
                feat_list.append(feat)
        return feat_list

    def status_features(self):
        """ Features related to stack/queue status
        """
        # Stack
        # yield ('Stack', 'Size', len(self.stack))
        # yield ('Queue', 'Size', len(self.queue))
        if (self.top1span is None) and (self.top2span is None):
            yield ('Stack', 'Empty')
        elif (self.top1span is not None) and (self.top2span is None):
            yield ('Stack', 'OneElem')
        elif len(self.stack) == 2:
            yield ('Stack', 'TwoElem')
        else:
            yield ('Stack', 'MoreElem')

        # Queue
        if self.firstspan is None:
            yield ('Queue', 'Empty')
        elif len(self.queue) == 1:
            yield ('Queue', 'OneElem')
        else:
            yield ('Queue', 'MoreElem')

    def operational_features(self):
        """
        features about the operations on stack and queue
        """
        # action history
        if len(self.action_hist) == 0:
            yield ('First action')
        if len(self.action_hist) > 0:
            yield ('Last action', action2str(self.action_hist[-1]))
            # if len(self.action_hist) > 1:
            # yield ('Last two action', action2str(self.action_hist[-1]), action2str(self.action_hist[-2]))

    def organizational_features(self):
        # ---------------------------------------
        # Whether within same sentence and paragraph
        # Span 1 and 2
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text1[0]].sidx == self.doc.token_dict[text2[-1]].sidx:
                yield ('Top12-Stack', 'SentContinue', True)
            else:
                yield ('Top12-Stack', 'SentContinue', False)
            if self.doc.token_dict[text1[0]].pidx == self.doc.token_dict[text2[-1]].pidx:
                yield ('Top12-Stack', 'ParaContinue', True)
            else:
                yield ('Top12-Stack', 'ParaContinue', False)
        # Span 1 and top span
        # First word from span 1, last word from span 3
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text3[0]].sidx:
                yield ('Stack-Queue', 'SentContinue', True)
            else:
                yield ('Stack-Queue', 'SentContinue', False)
            if self.doc.token_dict[text1[-1]].pidx == self.doc.token_dict[text3[0]].pidx:
                yield ('Stack-Queue', 'ParaContinue', True)
            else:
                yield ('Stack-Queue', 'ParaContinue', False)
        # # Last word from span 1, first word from span 2
        top12_stack_same_sent, top12_stack_same_para = False, False
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text2[0]].sidx:
                top12_stack_same_sent = True
                yield ('Top12-Stack', 'SameSent', True)
            else:
                yield ('Top12-Stack', 'SameSent', False)
            if self.doc.token_dict[text1[-1]].pidx == self.doc.token_dict[text2[0]].pidx:
                top12_stack_same_para = True
                yield ('Top12-Stack', 'SamePara', True)
            else:
                yield ('Top12-Stack', 'SamePara', False)
        # # Span 1 and top span
        # # First word from span 1, last word from span 3
        stack_queue_same_sent, stack_queue_same_para = False, False
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[0]].sidx == self.doc.token_dict[text3[-1]].sidx:
                stack_queue_same_sent = True
                yield ('Stack-Queue', 'SameSent', True)
            else:
                yield ('Stack-Queue', 'SameSent', False)
            if self.doc.token_dict[text1[0]].pidx == self.doc.token_dict[text3[-1]].pidx:
                stack_queue_same_para = True
                yield ('Stack-Queue', 'SamePara', True)
            else:
                yield ('Stack-Queue', 'SamePara', False)
        if top12_stack_same_sent and stack_queue_same_sent:
            yield ('Top12-Stack-Queue', 'SameSent', True)
        else:
            yield ('Top12-Stack-Queue', 'SameSent', False)
        if top12_stack_same_para and stack_queue_same_para:
            yield ('Top12-Stack-Queue', 'SamePara', True)
        else:
            yield ('Top12-Stack-Queue', 'SamePara', False)
        # ---------------------------------------
        # whether span is the start or end of sentence, paragraph or document
        if self.top1span is not None:
            text = self.top1span.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield ('Top1', 'Sent-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield ('Top1', 'Sent-end', True)
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield ('Top1', 'Para-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield ('Top1', 'Para-end', True)
            if text[0] - 1 < 0:
                yield ('Top1', 'Doc-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield ('Top1', 'Doc-end', True)
        if self.top2span is not None:
            text = self.top2span.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield ('Top2', 'Sent-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield ('Top2', 'Sent-end', True)
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield ('Top2', 'Para-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield ('Top2', 'Para-end', True)
            if text[0] - 1 < 0:
                yield ('Top2', 'Doc-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield ('Top2', 'Doc-end', True)
        if self.firstspan is not None:
            text = self.firstspan.text
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].sidx != self.doc.token_dict[text[0]].sidx:
                yield ('Queue', 'Sent-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].sidx != \
                    self.doc.token_dict[text[-1]].sidx:
                yield ('Queue', 'Sent-end', True)
            if text[0] - 1 < 0 or self.doc.token_dict[text[0] - 1].pidx != self.doc.token_dict[text[0]].pidx:
                yield ('Queue', 'Para-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict) or self.doc.token_dict[text[-1] + 1].pidx != \
                    self.doc.token_dict[text[-1]].pidx:
                yield ('Queue', 'Para-end', True)
            if text[0] - 1 < 0:
                yield ('Queue', 'Doc-start', True)
            if text[-1] + 1 >= len(self.doc.token_dict):
                yield ('Queue', 'Doc-end', True)

    def syntactic_featues(self):
        top12_stack_same_sent = False
        stack_queue_same_sent = False
        if self.top1span is not None and self.top2span is not None:
            text1, text2 = self.top1span.text, self.top2span.text
            if self.doc.token_dict[text1[-1]].sidx == self.doc.token_dict[text2[0]].sidx:
                top12_stack_same_sent = True
        if self.top1span is not None and self.firstspan is not None:
            text1, text3 = self.top1span.text, self.firstspan.text
            if self.doc.token_dict[text1[0]].sidx == self.doc.token_dict[text3[-1]].sidx:
                stack_queue_same_sent = True
        # syntactic dependency features
        if top12_stack_same_sent:
            text1, text2 = self.top1span.text, self.top2span.text
            text1_tidx = [self.doc.token_dict[token].tidx for token in text1]
            text1_heads = [self.doc.token_dict[token].hidx for token in text1]
            text1_deps = [self.doc.token_dict[token].dep_label for token in text1]
            text2_tidx = [self.doc.token_dict[token].tidx for token in text2]
            text2_heads = [self.doc.token_dict[token].hidx for token in text2]
            text2_deps = [self.doc.token_dict[token].dep_label for token in text2]
            right_dep, left_dep = False, False
            for idx, head in enumerate(text1_heads):
                if head in text2_tidx:
                    right_dep = True
                    yield ('Top12-Stack', 'Right-Dep', True)
                    yield ('Top12-Stack', 'Dep-Relation', text1_deps[idx])
                    # yield ('Top12-Stack', 'Right-Dep-Relation', text1_deps[idx])
                    # yield ('Top12-Stack', 'Right-Dep-Head', self.doc.token_dict[head-1].lemma)
                    break
            if not right_dep:
                for idx, head in enumerate(text2_heads):
                    if head in text1_tidx:
                        left_dep = True
                        yield ('Top12-Stack', 'Left-Dep', True)
                        yield ('Top12-Stack', 'Dep-Relation', text2_deps[idx])
                        # yield ('Top12-Stack', 'Left-Dep-Relation', text2_deps[idx])
                        # yield ('Top12-Stack', 'Left-Dep-Head', self.doc.token_dict[head - 1].lemma)
                        break
            if not right_dep and not left_dep:
                yield ('Top12-Stack', 'No-Dep')
        if stack_queue_same_sent:
            text1, text2 = self.top1span.text, self.firstspan.text
            text1_tidx = [self.doc.token_dict[token].tidx for token in text1]
            text1_heads = [self.doc.token_dict[token].hidx for token in text1]
            text1_deps = [self.doc.token_dict[token].dep_label for token in text1]
            text2_tidx = [self.doc.token_dict[token].tidx for token in text2]
            text2_heads = [self.doc.token_dict[token].hidx for token in text2]
            text2_deps = [self.doc.token_dict[token].dep_label for token in text2]
            right_dep, left_dep = False, False
            for idx, head in enumerate(text1_heads):
                if head in text2_tidx:
                    right_dep = True
                    yield ('Stack-Queue', 'Right-Dep', True)
                    yield ('Stack-Queue', 'Dep-Relation', text1_deps[idx])
                    break
            if not right_dep:
                for idx, head in enumerate(text2_heads):
                    if head in text1_tidx:
                        left_dep = True
                        yield ('Stack-Queue', 'Left-Dep', True)
                        yield ('Stack-Queue', 'Dep-Relation', text2_deps[idx])
                        break
            if not right_dep and not left_dep:
                yield ('Stack-Queue', 'No-Dep')

    def structural_features(self):
        # subtree form
        if self.top1span is not None and self.top1span.form is not None:
            yield ('Top1-Stack', 'Form', self.top1span.form)
        if self.top2span is not None and self.top2span.form is not None:
            yield ('Top2-Stack', 'Form', self.top2span.form)
        if self.top1span is not None and self.top2span is not None \
                and (self.top1span.form is not None or self.top2span.form is not None):
            yield ('Top12-Stack', 'Form', self.top1span.form, self.top2span.form)
        # distance
        if self.top1span is not None:
            dist_to_begin, dist_to_end = get_dist_to_begin_end(self.top1span, self.doc)
            if self.top1span.level == 0:
                yield ('Top1-Stack', 'Dist-To-Sent-Begin', dist_to_begin)
                yield ('Top1-Stack', 'Dist-To-Sent-End', dist_to_end)
            if self.top1span.level == 1:
                yield ('Top1-Stack', 'Dist-To-Para-Begin', dist_to_begin)
                yield ('Top1-Stack', 'Dist-To-Para-End', dist_to_end)
            if self.top1span.level == 2:
                yield ('Top1-Stack', 'Dist-To-Doc-Begin', dist_to_begin)
                yield ('Top1-Stack', 'Dist-To-Doc-End', dist_to_end)
        if self.top2span is not None:
            dist_to_begin, dist_to_end = get_dist_to_begin_end(self.top2span, self.doc)
            if self.top2span.level == 0:
                yield ('Top2-Stack', 'Dist-To-Sent-Begin', dist_to_begin)
                yield ('Top2-Stack', 'Dist-To-Sent-End', dist_to_end)
            if self.top2span.level == 1:
                yield ('Top2-Stack', 'Dist-To-Para-Begin', dist_to_begin)
                yield ('Top2-Stack', 'Dist-To-Para-End', dist_to_end)
            if self.top2span.level == 2:
                yield ('Top2-Stack', 'Dist-To-Doc-Begin', dist_to_begin)
                yield ('Top2-Stack', 'Dist-To-Doc-End', dist_to_end)
        if self.firstspan is not None:
            dist_to_begin, dist_to_end = get_dist_to_begin_end(self.firstspan, self.doc)
            if self.firstspan.level == 0:
                yield ('First-Queue', 'Dist-To-Sent-Begin', dist_to_begin)
                yield ('First-Queue', 'Dist-To-Sent-End', dist_to_end)
            if self.firstspan.level == 1:
                yield ('First-Queue', 'Dist-To-Para-Begin', dist_to_begin)
                yield ('First-Queue', 'Dist-To-Para-End', dist_to_end)
            if self.firstspan.level == 2:
                yield ('First-Queue', 'Dist-To-Doc-Begin', dist_to_begin)
                yield ('First-Queue', 'Dist-To-Doc-End', dist_to_end)

        # ---------------------------------------
        # EDU length
        if self.top1span is not None:
            edulen1 = self.top1span.edu_span[1] - self.top1span.edu_span[0] + 1
            yield ('Top1-Stack', 'nEDUs', categorize_length(edulen1))
        if self.top2span is not None:
            edulen2 = self.top2span.edu_span[1] - self.top2span.edu_span[0] + 1
            yield ('Top2-Stack', 'nEDUs', categorize_length(edulen2))
        if (self.top1span is not None) and (self.top2span is not None):
            if edulen1 > edulen2:
                yield ('Top-Stack', 'EDU-Comparison', True)
            elif edulen1 < edulen2:
                yield ('Top-Stack', 'EDU-Comparison', False)
            else:
                yield ('Top-Stack', 'EDU-Comparison', 'Equal')
        # ---------------------------------------
        # Sentence length
        if self.top1span is not None:
            text1 = self.top1span.text
            sentlen1 = self.doc.token_dict[text1[-1]].sidx - self.doc.token_dict[text1[0]].sidx + 1
            yield ('Top1-Stack', 'nSents', categorize_length(sentlen1))
        if self.top2span is not None:
            text2 = self.top2span.text
            sentlen2 = self.doc.token_dict[text2[-1]].sidx - self.doc.token_dict[text2[0]].sidx + 1
            yield ('Top2-Stack', 'nSents', categorize_length(sentlen2))
        if (self.top1span is not None) and (self.top2span is not None):
            if sentlen1 > sentlen2:
                yield ('Top-Stack', 'Sent-Comparison', True)
            elif sentlen1 < sentlen2:
                yield ('Top-Stack', 'Sent-Comparison', False)
            else:
                yield ('Top-Stack', 'Sent-Comparison', 'Equal')

                # ---------------------------------------
                # paragraph length
                # if self.top1span is not None:
                #     text1 = self.top1span.text
                #     paralen1 = self.doc.token_dict[text1[-1]].pidx - self.doc.token_dict[text1[0]].pidx + 1
                #     yield ('Top1-Stack', 'nParas', paralen1)
                # if self.top2span is not None:
                #     text2 = self.top2span.text
                #     paralen2 = self.doc.token_dict[text2[-1]].pidx - self.doc.token_dict[text2[0]].pidx + 1
                #     yield ('Top2-Stack', 'nParas', paralen2)
                # if (self.top1span is not None) and (self.top2span is not None):
                #     if paralen1 > paralen2:
                #         yield ('Top-Stack', 'Para-Comparison', True)
                #     elif paralen1 < paralen2:
                #         yield ('Top-Stack', 'Para-Comparison', False)
                #     else:
                #         yield ('Top-Stack', 'Para-Comparison', 'Equal')

    def ngram_features(self):
        """ Features about tokens in one text span
        """
        if self.top1span is not None:
            span = self.top1span
            # yield ('Top1-Stack', 'nTokens', len(span.text))
            # yield ('Top1-Stack', 'Word1_Suffix', get_suffix(self.doc.token_dict[span.text[0]].word))
            grams = get_grams(span.text, self.doc.token_dict)
            for gram in grams:
                yield ('Top1-Stack', 'nGram', gram)
        if self.top2span is not None:
            span = self.top2span
            # yield ('Top2-Stack', 'Word1_Suffix', get_suffix(self.doc.token_dict[span.text[0]].word))
            # yield ('Top2-Stack', 'nTokens', len(span.text))
            grams = get_grams(span.text, self.doc.token_dict)
            for gram in grams:
                yield ('Top2-Stack', 'nGram', gram)
        if self.firstspan is not None:
            span = self.firstspan
            # yield ('First-Queue', 'Word1_Suffix', get_suffix(self.doc.token_dict[span.text[0]].word))
            # yield ('First-Queue', 'nTokens', len(span.text))
            grams = get_grams(span.text, self.doc.token_dict)
            for gram in grams:
                yield ('First-Queue', 'nGram', gram)
        if self.top1span is not None and self.top2span is not None:
            span1 = self.top1span
            span2 = self.top2span
            # yield ('Top12-Stack', 'nTokens', len(span1.text)+len(span2.text))
            grams = get_conjunctive_grams(span2.text, span1.text, self.doc.token_dict)
            for gram in grams:
                yield ('Top12-Stack', 'nGram', gram)
        if self.top1span is not None and self.firstspan is not None:
            span1 = self.top1span
            span2 = self.firstspan
            # yield ('Stack-Queue', 'nTokens', len(span1.text)+len(span2.text))
            grams = get_conjunctive_grams(span1.text, span2.text, self.doc.token_dict)
            for gram in grams:
                yield ('Stack-Queue', 'nGram', gram)

    def nucleus_features(self):
        """ Feature extracted from one single nucleus EDU
        """
        for span_name, span in [('Top1', self.top1span), ('Top2', self.top2span), ('Queue', self.firstspan)]:
            if span is None:
                continue
            text = self.doc.edu_dict[span.nuc_edu]
            # for gidx in text:
            #     token = self.doc.token_dict[gidx]
            #     # yield (span_name, 'Nuc-word', token.lemma)
            #     yield (span_name, 'Nuc-pos', token.pos)
            text_tidx = [self.doc.token_dict[token].tidx for token in text]
            text_heads = [self.doc.token_dict[token].hidx for token in text]
            text_deps = [self.doc.token_dict[token].dep_label for token in text]
            for idx, head in enumerate(text_heads):
                if head not in text_tidx:
                    head_token = self.doc.token_dict[text_tidx[idx] - 1]
                    yield (span_name, 'Nuc-EDU-head-word', head_token.lemma)
                    yield (span_name, 'Nuc-EDU-head-pos', head_token.pos)
                    yield (span_name, 'Nuc-EDU-head-dep', text_deps[idx])

                    # if self.top1span is not None and self.top2span is not None:
                    #     yield ('Top12-Stack', 'Nuc-Edu-Dist', self.top1span.nuc_edu - self.top2span.nuc_edu)
                    # if self.top1span is not None and self.firstspan is not None:
                    #     yield ('Stack-Queue', 'Nuc-Edu-Dist', self.firstspan.nuc_edu - self.top1span.nuc_edu)

    def bc_features(self):
        """ Feature extract from brown clusters
            Features are only extracted from Nucleus EDU !!!!
        """
        token_dict = self.doc.token_dict
        edu_dict = self.doc.edu_dict
        if self.top1span is not None:
            eduidx = self.top1span.nuc_edu
            bcfeatures = get_bc(eduidx, edu_dict, token_dict, self.bcvocab, self.npref)
            for feat in bcfeatures:
                yield ('BC', 'Top1Span', feat)
        if self.top2span is not None:
            eduidx = self.top2span.nuc_edu
            bcfeatures = get_bc(eduidx, edu_dict, token_dict, self.bcvocab, self.npref)
            for feat in bcfeatures:
                yield ('BC', 'Top2Span', feat)
        if self.firstspan is not None:
            eduidx = self.firstspan.nuc_edu
            bcfeatures = get_bc(eduidx, edu_dict, token_dict, self.bcvocab, self.npref)
            for feat in bcfeatures:
                yield ('BC', 'FirstSpan', feat)


class RelationFeatureGenerator(object):
    def __init__(self, node, rst_tree, level, bcvocab, nprefix=11):
        self.level = level
        self.node = node
        self.lnode = self.node.lnode
        self.rnode = self.node.rnode
        self.pnode = self.node.pnode
        self.rst_tree = rst_tree
        self.doc = self.rst_tree.doc
        self.root = self.rst_tree.tree
        self.bcvocab = bcvocab
        self.nprefix = nprefix
        # Doc length wrt EDUs
        self.doclen = len(self.doc.edu_dict)

    def gen_features(self):
        """ Main function to generate features
                """
        feat_list = []
        for feat in self.lexical_features():
            feat_list.append(feat)

        for feat in self.structural_features():
            feat_list.append(feat)

        if self.level in [1, 2]:
            for feat in self.form_features():
                feat_list.append(feat)

        if self.level in [1, 2]:
            for feat in self.tree_features():
                feat_list.append(feat)

        if self.level in [1, 2]:
            for feat in self.nucleus_features():
                feat_list.append(feat)
        if self.level in [0]:
            for feat in self.syntactic_features():
                feat_list.append(feat)
        # Brown clusters
        if self.level in [1, 2] and self.bcvocab is not None:
            for feat in self.bc_features():
                feat_list.append(feat)
        return feat_list

    def lexical_features(self):
        left_text, right_text = self.lnode.text, self.rnode.text
        # yield ('Lnode', 'nTokens', len(left_text))
        # yield ('Lnode', 'Word1_Suffix', get_suffix(self.doc.token_dict[left_text[0]].word))
        grams = get_grams(left_text, self.doc.token_dict)
        for gram in grams:
            yield ('Lnode', 'nGram', gram)
        # yield ('Rnode', 'nTokens', len(right_text))
        # yield ('Rnode', 'Word1_Suffix', get_suffix(self.doc.token_dict[right_text[0]].word))
        grams = get_grams(right_text, self.doc.token_dict)
        for gram in grams:
            yield ('Rnode', 'nGram', gram)
        # yield ('LRnode', 'nTokens', len(left_text) + len(right_text))
        # yield ('LRnode', 'Word2_Suffix', get_suffix(self.doc.token_dict[left_text[0]].word), get_suffix(self.doc.token_dict[right_text[0]].word))
        grams = get_conjunctive_grams(left_text, right_text, self.doc.token_dict)
        for gram in grams:
            yield ('LRnode', 'nGram', gram)

    def syntactic_features(self):
        left_text, right_text = self.lnode.text, self.rnode.text
        left_text_tidx = [self.doc.token_dict[token].tidx for token in left_text]
        left_text_heads = [self.doc.token_dict[token].hidx for token in left_text]
        left_text_deps = [self.doc.token_dict[token].dep_label for token in left_text]
        right_text_tidx = [self.doc.token_dict[token].tidx for token in right_text]
        right_text_heads = [self.doc.token_dict[token].hidx for token in right_text]
        right_text_deps = [self.doc.token_dict[token].dep_label for token in right_text]
        right_dep, left_dep = False, False
        for idx, head in enumerate(left_text_heads):
            if head in right_text_tidx:
                right_dep = True
                yield ('LRnode', 'Right-Dep', True)
                yield ('LRnode', 'Dep-Relation', left_text_deps[idx])
                break
        if not right_dep:
            for idx, head in enumerate(right_text_heads):
                if head in left_text_tidx:
                    left_dep = True
                    yield ('LRnode', 'Left-Dep', True)
                    yield ('LRnode', 'Dep-Relation', right_text_deps[idx])
                    break
        if not right_dep and not left_dep:
            yield ('LRnode', 'No-Dep')

    def structural_features(self):
        if self.node is not None:
            dist_to_begin, dist_to_end = get_dist_to_begin_end(self.node, self.doc)
            if self.node.level == 0:
                yield ('Self', 'Dist-To-Sent-Begin', dist_to_begin)
                yield ('Self', 'Dist-To-Sent-End', dist_to_end)
            if self.node.level == 1:
                yield ('Self', 'Dist-To-Para-Begin', dist_to_begin)
                yield ('Self', 'Dist-To-Para-End', dist_to_end)
            if self.node.level == 2:
                yield ('Self', 'Dist-To-Doc-Begin', dist_to_begin)
                yield ('Self', 'Dist-To-Doc-End', dist_to_end)
        if self.lnode is not None:
            dist_to_begin, dist_to_end = get_dist_to_begin_end(self.lnode, self.doc)
            if self.lnode.level == 0:
                yield ('Lnode', 'Dist-To-Sent-Begin', dist_to_begin)
                yield ('Lnode', 'Dist-To-Sent-End', dist_to_end)
            if self.lnode.level == 1:
                yield ('Lnode', 'Dist-To-Para-Begin', dist_to_begin)
                yield ('Lnode', 'Dist-To-Para-End', dist_to_end)
            if self.lnode.level == 2:
                yield ('Lnode', 'Dist-To-Doc-Begin', dist_to_begin)
                yield ('Lnode', 'Dist-To-Doc-End', dist_to_end)
        if self.rnode is not None:
            dist_to_begin, dist_to_end = get_dist_to_begin_end(self.rnode, self.doc)
            if self.rnode.level == 0:
                yield ('Rnode', 'Dist-To-Sent-Begin', dist_to_begin)
                yield ('Rnode', 'Dist-To-Sent-End', dist_to_end)
            if self.rnode.level == 1:
                yield ('Rnode', 'Dist-To-Para-Begin', dist_to_begin)
                yield ('Rnode', 'Dist-To-Para-End', dist_to_end)
            if self.rnode.level == 2:
                yield ('Rnode', 'Dist-To-Doc-Begin', dist_to_begin)
                yield ('Rnode', 'Dist-To-Doc-End', dist_to_end)

        if self.level == 0:
            # ---------------------------------------
            # EDU length
            if self.lnode is not None:
                edulen1 = self.lnode.edu_span[1] - self.lnode.edu_span[0] + 1
                yield ('Lnode', 'nEDUs', categorize_length(edulen1))
            if self.rnode is not None:
                edulen2 = self.rnode.edu_span[1] - self.rnode.edu_span[0] + 1
                yield ('Rnode', 'nEDUs', categorize_length(edulen2))
            if (self.lnode is not None) and (self.rnode is not None):
                # yield ('LRnode', 'Edu-Diff', edulen1 - edulen2)
                if edulen1 > edulen2:
                    yield ('LRnode', 'EDU-Comparison', True)
                elif edulen1 < edulen2:
                    yield ('LRnode', 'EDU-Comparison', False)
                else:
                    yield ('LRnode', 'EDU-Comparison', 'Equal')

        if self.level == 1:
            # ---------------------------------------
            # sentence length
            if self.lnode is not None:
                text1 = self.lnode.text
                sentlen1 = self.doc.token_dict[text1[-1]].sidx - self.doc.token_dict[text1[0]].sidx + 1
                yield ('Lnode', 'nSents', categorize_length(sentlen1))
            if self.rnode is not None:
                text2 = self.rnode.text
                sentlen2 = self.doc.token_dict[text2[-1]].sidx - self.doc.token_dict[text2[0]].sidx + 1
                yield ('Rnode', 'nSents', categorize_length(sentlen2))
            if (self.lnode is not None) and (self.rnode is not None):
                # yield ('LRnode', 'Sent-Diff', sentlen1 - sentlen2)
                if sentlen1 > sentlen2:
                    yield ('LRnode', 'Sent-Comparison', True)
                elif sentlen1 < sentlen2:
                    yield ('LRnode', 'Sent-Comparison', False)
                else:
                    yield ('LRnode', 'Sent-Comparison', 'Equal')

        if self.level == 2:
            # ---------------------------------------
            # paragraph length
            if self.lnode is not None:
                text1 = self.lnode.text
                paralen1 = self.doc.token_dict[text1[-1]].pidx - self.doc.token_dict[text1[0]].pidx + 1
                yield ('Lnode', 'nParas', categorize_length(paralen1))
            if self.rnode is not None:
                text2 = self.rnode.text
                paralen2 = self.doc.token_dict[text2[-1]].pidx - self.doc.token_dict[text2[0]].pidx + 1
                yield ('Rnode', 'nParas', categorize_length(paralen2))
            if (self.lnode is not None) and (self.rnode is not None):
                # yield ('LRnode', 'Para-Diff', paralen1 - paralen2)
                if paralen1 > paralen2:
                    yield ('LRnode', 'Para-Comparison', True)
                elif paralen1 < paralen2:
                    yield ('LRnode', 'Para-Comparison', False)
                else:
                    yield ('LRnode', 'Para-Comparison', 'Equal')

    def form_features(self):
        # form
        yield ('Self', 'Form', self.node.form)
        if self.node.pnode is not None:
            yield ('Pnode', 'Form', self.node.form)
        # if self.lnode is not None and self.lnode.form is not None:
        #     yield ('Lnode', 'Form', self.lnode.form)
        # if self.rnode is not None and self.rnode.form is not None:
        #     yield ('Rnode', 'Form', self.rnode.form)
        # if self.lnode is not None and self.rnode is not None \
        #         and (self.lnode.form is not None or self.rnode.form is not None):
        #     yield ('LRnode', 'Form', self.lnode.form, self.rnode.form)
        # prop
        yield ('Self', 'Prop', self.node.prop)
        if self.node.pnode is not None:
            yield ('Pnode', 'Prop', self.node.pnode.prop)

    def tree_features(self):
        # depth
        yield ('Self', 'Depth', self.node.depth)
        # if self.node.level == 2:
        #     yield ('Self', 'Doc-Depth', self.node.depth)
        #     yield ('Self', 'Max-Depth', self.node.max_depth)
        # if self.node.level == 1:
        #     para_root = self.node.pnode
        #     child_node = self.node
        #     while para_root is not None and para_root.level == 1:
        #         child_node = para_root
        #         para_root = para_root.pnode
        #     yield ('Self', 'Paragraph -Depth', self.node.depth - child_node.depth)
        #     yield ('Self', 'Paragraph-Max-Depth', self.node.max_depth - child_node.depth)
        # yield ('Self', 'Max-Depth', self.node.max_depth)
        # yield ('Lnode', 'Max-Depth', self.lnode.max_depth)
        # yield ('Rnode', 'Max-Depth', self.rnode.max_depth)
        # yield ('LRnode', 'Max-Depth-diff', self.lnode.max_depth - self.rnode.max_depth)
        # height
        yield ('Self', 'Height', categorize_percent(self.node.height / self.root.height))
        # yield ('Lnode', 'Height', categorize_percent(self.lnode.height / self.root.height))
        # yield ('Rnode', 'Height', categorize_percent(self.rnode.height / self.root.height))
        # yield ('LRnode', 'Height-diff', categorize_percent((self.lnode.height - self.rnode.height) / self.root.height))

    def nucleus_features(self):
        """ Feature extracted from one single nucleus EDU
        """
        for span_name, span in [('Lnode', self.lnode), ('Rnode', self.rnode)]:
            if span is None:
                continue
            text = self.doc.edu_dict[span.nuc_edu]
            # for gidx in text:
            #     token = self.doc.token_dict[gidx]
            #     # yield (span_name, 'Nuc-word', token.lemma)
            #     yield (span_name, 'Nuc-pos', token.pos)
            text_tidx = [self.doc.token_dict[token].tidx for token in text]
            text_heads = [self.doc.token_dict[token].hidx for token in text]
            text_deps = [self.doc.token_dict[token].dep_label for token in text]
            for idx, head in enumerate(text_heads):
                if head not in text_tidx:
                    head_token = self.doc.token_dict[text_tidx[idx] - 1]
                    yield (span_name, 'Nuc-EDU-head-word', head_token.lemma)
                    yield (span_name, 'Nuc-EDU-head-pos', head_token.pos)
                    yield (span_name, 'Nuc-EDU-head-dep', text_deps[idx])

    def bc_features(self):
        """ Feature extract from brown clusters
            Features are only extracted from Nucleus EDU !!!!
        """
        token_dict = self.doc.token_dict
        edu_dict = self.doc.edu_dict
        if self.lnode is not None:
            eduidx = self.lnode.nuc_edu
            bcfeatures = get_bc(eduidx, edu_dict, token_dict, self.bcvocab, self.nprefix)
            for feat in bcfeatures:
                yield ('BC', 'Lnode', feat)
        if self.rnode is not None:
            eduidx = self.rnode.nuc_edu
            bcfeatures = get_bc(eduidx, edu_dict, token_dict, self.bcvocab, self.nprefix)
            for feat in bcfeatures:
                yield ('BC', 'Rnode', feat)


def get_grams(text, token_dict):
    """ Generate first one, two words from the token list

    :type text: list of int
    :param text: indices of words with the text span

    :type token_dict: dict of Token (data structure)
    :param token_dict: all tokens in the doc, indexing by the
                      document-level index
    """
    n = len(text)
    grams = set()
    # Get lower-case of words
    if n >= 1:
        grams.add(('Start-Unigram-Word', token_dict[text[0]].lemma.lower()))
        grams.add(('End-Unigram-Word', token_dict[text[-1]].lemma.lower()))
        grams.add(('Start-Unigram-Pos', token_dict[text[0]].pos.lower()))
        grams.add(('End-Unigram-Pos', token_dict[text[-1]].pos.lower()))
    if n >= 2:
        token = token_dict[text[0]].lemma.lower() + ' ' + token_dict[text[1]].lemma.lower()
        grams.add(('Start-Bigram-Word', token))
        token = token_dict[text[0]].pos.lower() + ' ' + token_dict[text[1]].pos.lower()
        grams.add(('Start-Bigram-Pos', token))
        token = token_dict[text[-2]].lemma.lower() + ' ' + token_dict[text[-1]].lemma.lower()
        grams.add(('End-Bigram-Word', token))
        token = token_dict[text[-2]].pos.lower() + ' ' + token_dict[text[-1]].pos.lower()
        grams.add(('End-Bigram-Pos', token))
    return grams


def get_conjunctive_grams(text1, text2, token_dict):
    """
    Generate conjunctive 2-grams for continuous spans
    :param text1:
    :param text2:
    :param token_dict:
    :return:
    """
    n1 = len(text1)
    n2 = len(text2)
    grams = set()
    if n1 > 0 and n2 > 0:
        grams.add(('Conjunctive-Word', token_dict[text1[0]].lemma.lower() + ' ' + token_dict[text2[0]].lemma.lower()))
        grams.add(('Conjunctive-Pos', token_dict[text1[0]].pos.lower() + ' ' + token_dict[text2[0]].pos.lower()))
    # if n1 > 1 and n2 > 0:
    #     grams.add(token_dict[text1[-1]].lemma.lower() + ' ' + token_dict[text2[0]].lemma.lower())
    return grams


def get_suffix(word):
    suffix_set = {'ing', 'ed', 'ly'}
    for suffix in suffix_set:
        if word.endswith(suffix):
            return suffix
    return None


def get_dist_to_begin_end(node, doc):
    dist_to_begin = -1
    dist_to_end = -1
    if node.level == 0:
        sent_idx = doc.token_dict[node.text[0]].sidx
        sent_start_tidx = node.text[0]
        while sent_start_tidx >= 0 and doc.token_dict[sent_start_tidx].sidx == sent_idx:
            sent_start_tidx -= 1
        sent_start_tidx += 1
        dist_to_begin = doc.token_dict[node.text[0]].eduidx - doc.token_dict[sent_start_tidx].eduidx
        sent_end_tidx = node.text[-1]
        while sent_end_tidx < len(doc.token_dict) and doc.token_dict[sent_end_tidx].sidx == sent_idx:
            sent_end_tidx += 1
        sent_end_tidx -= 1
        dist_to_end = doc.token_dict[sent_end_tidx].eduidx - doc.token_dict[node.text[-1]].eduidx

    if node.level == 1 and node.lnode is not None and node.rnode is not None:
        para_idx = doc.token_dict[node.text[0]].pidx
        para_start_tidx = node.text[0]
        while para_start_tidx >= 0 and doc.token_dict[para_start_tidx].pidx == para_idx:
            para_start_tidx -= 1
        para_start_tidx += 1
        dist_to_begin = doc.token_dict[node.text[0]].sidx - doc.token_dict[para_start_tidx].sidx
        para_end_tidx = node.text[-1]
        while para_end_tidx < len(doc.token_dict) and doc.token_dict[para_end_tidx].pidx == para_idx:
            para_end_tidx += 1
        para_end_tidx -= 1
        dist_to_end = doc.token_dict[para_end_tidx].sidx - doc.token_dict[node.text[-1]].sidx

    if node.level == 2:
        dist_to_begin = doc.token_dict[node.text[0]].pidx
        dist_to_end = doc.token_dict[len(doc.token_dict) - 1].pidx - doc.token_dict[node.text[-1]].pidx

    return dist_to_begin, dist_to_end


def get_bc(eduidx, edu_dict, token_dict, bcvocab, nprefix=5):
    """ Get brown cluster features for tokens

    :type eduidx: int
    :param eduidx: index of one EDU

    :type edu_dict: dict
    :param edu_dict: All EDUs in one dict

    :type token_dict: dict of Token (data structure)
    :param token_dict: all tokens in the doc, indexing by the
                      document-level index

    :type bcvocab: dict {word : braown-cluster-index}
    :param bcvocab: brown clusters

    :type nprefix: int
    :param nprefix: number of prefix we want to keep from
                    cluster indices
    """
    text = edu_dict[eduidx]
    bc_features = set()
    for gidx in text:
        tok = token_dict[gidx].lemma.lower()
        try:
            bc_idx = bcvocab[tok][:nprefix]
            bc_features.add(bc_idx)
        except KeyError:
            pass
    return bc_features


def categorize_length(length):
    if length < 1:
        return 0
    if length < 2:
        return 1
    if length < 4:
        return 2
    if length < 8:
        return 3
    if length < 16:
        return 4
    if length < 32:
        return 5
    if length < 64:
        return 6
    else:
        return 7


def categorize_percent(percent):
    return round(percent * 10)
