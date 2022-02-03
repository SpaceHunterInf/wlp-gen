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
        self.parent = []
        self.root = False
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
        node.add_parent(self)
    
    def add_parent(self, node):
        assert isinstance(node, Tree)
        self.parent.append(node)
    
    def set_root(self):
        self.root = True

    def get_inside(self, max = 3):
        batch = [self]
        res = []
        
        for i in range(max):
            lv = []
            for item in batch:
                res = res + item.children
                lv = lv + item.children
            batch = lv
        
        ans = []
        for item in res:
            ans.append(item.type)
            
        return ans

    def get_outside(self, max = 3):
        parent_batch = self.parent
        children_batch = []
        if self.root == True:
            return []
        for child in self.parent[0].children:
            if child.name != self.name:
                children_batch.append(child)
        
        res = []

        for i in range(max):
            parent_lv = []
            children_lv = []
            for item in parent_batch:
                res = res + item.parent
                parent_lv = parent_lv + item.parent
            for item in children_batch:
                res = res + item.children
                children_lv = children_lv + item.children
            parent_batch = parent_lv
            children_batch = children_lv
        
        ans = []
        for item in res:
            ans.append(item.type)

        return ans