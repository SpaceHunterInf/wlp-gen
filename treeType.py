from random import choice
import pydot

def draw_edge(parent, child, graph):
    edge = pydot.Edge(parent, child)
    graph.add_edge(edge)

def draw_tree(node):
    root = node
    graph = pydot.Dot(graph_type='graph')
    batch = [root]
    lv=0
    while batch != []:
        next_lvl = []
        for var in batch:
            if var.children != []:
                for child in var.children:
                    draw_edge(var.name, child.name, graph)
                    next_lvl.append(child)
        batch = next_lvl
        print(lv)
        for i in batch:
            print(i.name)
        lv=lv+1

    graph.write_png('example1_graph.png')
        


class Tree(object):
    "Generic tree node."
    def __init__(self, type='root', children=None):
        self.type = type
        self.prefix = '0'
        self.suffix = '0'
        self.name = type
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def set_prefix(self, name):
        self.prefix = name
        self.name = self.prefix + '-' + self.type + '-' + self.suffix

    def set_suffix(self, name):
        self.suffix = name
        self.name = self.prefix + '-' + self.type + '-' + self.suffix
        
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)