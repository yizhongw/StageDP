#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午8:49

class SpanNode(object):
    """ RST tree node
    """

    def __init__(self, prop):
        """ Initialization of SpanNode

        :type prop: string or None
        :param prop: property of this span
        """
        # Text of this span / Discourse relation
        self.text, self.relation = None, None
        # EDU span / Nucleus span (begin, end) index
        self.edu_span, self.nuc_span = None, None
        # Nucleus single EDU
        self.nuc_edu = None
        # Property
        self.prop = prop
        # Children node
        # Each of them is a node instance
        # N-S form (for binary RST tree only)
        self.lnode, self.rnode = None, None
        # Parent node
        self.pnode = None
        # Node list (for general RST tree only)
        self.nodelist = []
        # Relation form: NN, NS, SN
        self.form = None
        # Relation between its left child and right child
        self.child_relation = None
        # Depth of this node on RST tree
        self.depth = -1
        # Max depth of its subtree
        self.max_depth = -1
        # Height of this node on RST tree
        self.height = 0
        # level of this node, 0 for inner-sentence, 1 for inter-sentence but inner paragraph, 2 for inter-paragraph
        self.level = 0

    def create_node(self, content):
        """ Assign value to an SpanNode instance

        :type content: list
        :param content: content from stack
        """
        for c in content:
            if isinstance(c, SpanNode):
                # Sub-node
                self.nodelist.append(c)
                c.pnode = self
            elif c[0] == 'span':
                self.edu_span = (c[1], c[2])
            elif c[0] == 'relation':
                self.relation = c[1]
            elif c[0] == 'leaf':
                self.edu_span = (c[1], c[1])
                self.nuc_span = (c[1], c[1])
                self.nuc_edu = c[1]
            elif c[0] == 'text':
                self.text = c[1]
            else:
                raise ValueError("Unrecognized property: {}".format(c[0]))

    def assign_relation(self, relation):
        if self.form == 'NN':
            self.lnode.relation = relation
            self.rnode.relation = relation
        elif self.form == 'NS':
            self.lnode.relation = "span"
            self.rnode.relation = relation
        elif self.form == 'SN':
            self.lnode.relation = relation
            self.rnode.relation = "span"
        else:
            raise ValueError("Error when assign relation to node with form: {}".format(form))
