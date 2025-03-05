# coding: utf-8

import os

import networkx as nx
import graphviz as gv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from flexir.utilities.logger import logger, ASSERT

class NetGraph(object):
    def __init__(self):
        node_attr = dict(
            style=      'filled', \
            align=      'left', \
            fontsize=   '24', \
            ranksep=    '0.1', \
            height=     '0.1', \
            shape =     'box' \
            )
        self.GViz = gv.Digraph(format='png',node_attr=node_attr, graph_attr=dict(size='''{},{}'''.format(320, 320)))
        self.G = nx.DiGraph()
        self.nodes = {}

    def add_node(self, node, infos):
        self.GViz.node(name=node, label=infos)
        self.G.add_node(node)
        if node in self.nodes.keys():
            self.nodes[node]['infos'] = infos
        else:
            self.nodes[node] = { 'infos':infos, 'from':[], 'to':[] }

    def add_edge(self, from_node, to_node):
        self.GViz.edge(from_node, to_node)
        self.G.add_edge(from_node, to_node)
        self.nodes[to_node]['from'].append(from_node)
        self.nodes[from_node]['to'].append(to_node)

    def draw(self, path = None):
        self.GViz.render(path, cleanup=True)
        img = mpimg.imread(path+'.png', 1)
        imgplot = plt.imshow(img)
        plt.show()

    def topologicalsort(self):
        sorted_node = nx.topological_sort(self.G)
        return list(sorted_node)

    def save(self, name, outputDir):
        try:
            self.draw(os.path.join(outputDir, name))
        except Exception as e:
            logger.warning('[EXCEPTION] {}', e)
            logger.warning('[buildNetwork] Install graphviz dot command to draw png.')
