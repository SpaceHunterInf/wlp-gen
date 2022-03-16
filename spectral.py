import pickle
import numpy as np
from scipy.sparse.linalg import svds
from treeType import *
from utils import *
from tqdm import tqdm
from evaluator import *
from gen import *
import sys
import json

def t_svd(inside, outside, k):
    sigma = np.outer(inside, outside)
    u, s, vt = svds(sigma, k=min(min(sigma.shape) - 1, k))
    return u, s, vt

def svd_info(data, rules, types, in_max, out_max, k):
    #calculating projection matrix for all possible types
    res = {}
    type_matrix = {}
    for type in types:
        type_matrix[type]['count'] = 0
        type_matrix[type]['sigma'] = np.zeros((in_max, out_max))

    for tree in data:
        nodes = [tree]
        batch = [tree]
        while batch != []:
            for node in batch:
                lv = []
                lv = lv + node.children
                nodes = nodes + lv
                batch = lv
        #extracting all nodes in a tree
        
        for node in nodes:
            inside = feature_gen(node.get_inside(), rules, in_max) 
            outside = feature_gen(node.get_outside(), rules, out_max)
            type_matrix[node.type]['sigma'] = type_matrix[node.type]['sigma'] + np.outer(inside, outside)
            type_matrix[node.type]['count'] = type_matrix[node.type]['count'] + 1
    
    for type in types:
        u, s, vt = svds(type_matrix[type]['sigma'] / type_matrix[type]['count'], k=min(type_matrix[type]['sigma'] - 1, k))
        res[type]['u'] = u
        res[type]['s'] = s
        res[type]['vt'] = vt
    

    #I NEED TO FLIP THE SIGN OF THE NEGATIVE PROBABILITIES
    return res

def opt_svd(data, types, obs, in_max, out_max, max_k):
        res = {}
        type_matrix = {}
        for type in types:
            type_matrix[type] = {'count':0, 'sigma':np.zeros((in_max, out_max))}
            res[type] = {'u':0, 's':0, 'vt':0, 'in':None, 'out':None}
    
        for tree in data:
            nodes = [tree]
            batch = [tree]
            while batch != []:
                for node in batch:
                    lv = []
                    for child in node.children:
                        if child.type.isupper() != True:
                            lv.append(child)
                    nodes = nodes + lv
                    batch = lv
            #extracting all nodes in a tree
            for node in nodes:
                inside = feature_gen(node.get_inside(), types+obs, in_max) 
                outside = feature_gen(node.get_outside(), types+obs, out_max)
                type_matrix[node.type]['sigma'] = type_matrix[node.type]['sigma'] + np.outer(inside, outside)
                type_matrix[node.type]['count'] = type_matrix[node.type]['count'] + 1

        for type in types:
            # we try to find the optimal k here
            k = max_k
            while k >= 1:
                if type_matrix[type]['count'] > 0:
                    u, s, vt = svds(type_matrix[type]['sigma'] / type_matrix[type]['count'],k)
                    res[type]['u'] = u
                    res[type]['s'] = s
                    res[type]['vt'] = vt
                k = k - 1
                if not (0 in s):
                    break
        
        bias = 0.01

        for type in types:
            if 0 in res[type]['s']:
                res[type]['s'] = res[type]['s'] + bias
            
            res[type]['in_project'] = res[type]['u']
            res[type]['out_project'] = np.reciprocal(res[type]['s']).reshape(-1,1) * res[type]['vt']
        

        with open('res.pkl','wb') as f:
            pickle.dump(res,f)
        with open('mat.pkl','wb') as f:
            pickle.dump(type_matrix,f)

        return res, type_matrix

class spectral(object):
    def __init__(self, data, program, in_max, out_max, max_K):
        # data is the input of M training examples, a list of proof trees
        # program is the L-WLP only contains types and rules
        self.data = data
        self.program = program
        self.inmax = in_max
        self.outmax = out_max
        self.k = max_K
        self.add_one() # perform add 1 smoothing
        self.res, self.mat = opt_svd(data, program.types, program.obs_facts, in_max, out_max, max_K)

        # with open('res.pkl','rb') as f:
        #     self.res = pickle.load(f)
        # with open('mat.pkl','rb') as f:
        #     self.mat = pickle.load(f)

    def add_one(self):
        #basic add one smoothing
        for rule in self.program.rules:
            root = Tree(rule['parent'])
            root.set_root()
            for child in rule['children']:
                root.add_child(Tree(child))
            self.data.append(root)

    def param_estimate(self):
        relations = []
        for rule in self.program.rules:
            current = {'parent':[rule['parent']], 'children':rule['children'], 'param': None, 'count':0}
            shapeT = tuple([len(self.res[x]['s']) for x in current['parent']] + [len(self.res[x]['s']) for x in current['children'] if x.isupper()==False])
            current['param'] = np.zeros(shapeT)
            relations.append(current)
        # initialize all transition parameters to 0 based on latent states we've learned
        
        for tree in self.data:
            batch = [tree]
            while batch != []:
                lv = []
                for tree in batch:
                    tmp_parent = [tree]
                    tmp_parent_rule = [x.type for x in tmp_parent]
                    tmp_children = [x for x in tree.children] #if x.type.isupper() == False]
                    tmp_children_rule = [x.type for x in tmp_children]

                    for rule in relations:
                        if rule_eq(tmp_parent_rule, rule['parent'], tmp_children_rule, rule['children']):
                            for p in tmp_parent:
                                out_projected = feature_gen(p.get_outside(), self.program.types + self.program.obs_facts, self.outmax).dot(self.res[p.type]['out_project'].T)
                            
                            aligned_children = []
                            
                            for c1_type in rule['children']:
                                for c2 in tmp_children:
                                    if c2.type == c1_type and not(c2 in aligned_children):
                                        aligned_children.append(c2)
                                        break
                            
                            if all([c.isupper() for c in rule['children']]):
                                pass
                            else:
                                for c in aligned_children:
                                    #print(c.type)
                                    in_projected = feature_gen(c.get_inside(), self.program.types + self.program.obs_facts, self.inmax).dot(self.res[c.type]['in_project'])
                                    out_projected = np.multiply.outer(out_projected, in_projected)
                            
                            if rule['param'].shape != out_projected.shape:
                                #print(rule['parent'])
                                #print(rule['children'])
                                print(rule['children'])
                                print(aligned_children)
                                print(rule['param'].shape)
                                print(out_projected.shape)
                            rule['param'] = rule['param'] + out_projected
                            rule['count'] = rule['count'] + 1
                    lv = lv + tmp_children
                batch = lv
        
        for rule in relations:
            total_count = 0
            for p in rule['parent']:
                #print(p)
                total_count = total_count + self.mat[p]['count']
            rule['param'] = rule['param'] / total_count


        #get root probabilities
        root_rules = []
        types = []
        for tree in self.data:
            if tree.root:
                types.append(tree.type)
        
        for i in set(types):
            param = np.zeros(tuple([len(self.res[i]['s'])]))
            count = 0
            for tree in self.data:
                if tree.type == i:
                    in_projected = feature_gen(tree.get_inside(), self.program.types + self.program.obs_facts, self.inmax).dot(self.res[tree.type]['in_project'])
                    param = param + in_projected
                    count = count + 1
            
            rule = {'parent':[i], 'children':['root'], 'param':param / len(self.data), 'count':count}
            root_rules.append(rule)

        relations = relations + root_rules

        return relations

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    results = {}
    for i in range(5):
        with open('data-{}.pkl'.format(i), 'rb') as f:
            data = pickle.load(f)

        spec = spectral(data['train'], data['program'], 1000, 2000, 5)
        parameters = spec.param_estimate()
        root_parameters = [x for x in parameters if x['children'] == ['root']]
        test = evaluator(parameters, parameters, data['test'], get_best_parse)
        print('data-{}'.format(i))
        test_abs = evaluator(parameters, parameters, data['test'], get_best_parse_abs)
        results[i] = [test.micro_avg(), test.macro_avg(), test_abs.micro_avg(), test_abs.macro_avg()]

    with open('out_spectral.txt', 'w') as f:
        json.dump(results, f)

