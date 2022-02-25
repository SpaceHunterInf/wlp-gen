import imp
from treeType import *
import numpy as np
import copy
import collections


def make_onehot(type, total_types):
    res = np.zeros(len(total_types))
    res[total_types.index(type)] = 1
    # # is only used for padding
    return res

def feature_gen(types,total_types, max_len):
    res = []
    for i in types:
        res.append(make_onehot(i, total_types))
    
    res = np.array(res).flatten()
    if len(res >= max_len):
        res = res[:max_len]
    
    padding = np.zeros(max_len-len(res))

    #print(len(res))
    return np.concatenate((res,padding))
    
def inside_sweep(parent,param):
    parent_type = [parent]
    char = 'abcdefghijklmno'
    for rule in param:
        if parent in rule['parent']:
            if all([x.isupper() for x in rule['children']]):
                return rule['param']
            else:
                children_type = rule['children']
                transition = rule['param']
                formula = char[:len(parent_type)] + char[len(parent_type):len(parent_type) + len(children_type)] #shape of transiton
                head = formula
                for l in range(len(children_type)):
                    formula = formula + ',' + char[len(parent_type) + l] # shape of individual vector
                formula = formula + '->' +  char[:len(parent_type)]
                #print(formula)
                
                args = [formula, rule['param']]
                for c in rule['children']:
                    if c != parent:
                        #print(c)
                        #print(parent)
                        if not c.isupper():
                            args.append(inside_sweep(c, param))
                    else:
                        args.append(np.einsum(head+'->'+char[:len(parent_type)], rule['param']))
                #print(args)
                #print(parent)
                return np.einsum(*args)

def outside_sweep(parent, param):
    char = 'abcdefghijklmno'
    for rule in param:
        if rule['parent'] == [parent] and rule['children'] == ['root']:
            return rule['param']
    else:
        for rule in param:
            if parent in rule['children']:
                outside = outside_sweep(rule['parent'], param)
                
                formula = char[:len(rule['parent'])] + char[len(rule['parent']): len(rule['parent']) + len(rule['children'])]
                head = formula
                for l in range(len(rule['children'])):
                    formula = formula + ',' + char[len(rule['parent']) + l] # shape of individual vector
                formula = formula + '->' + char[len(rule['parent']) + rule['children'].index(parent)]
                
                #print(formula)
                args = [formula, rule['param']]
                for c in rule['children']:
                    #print(c)
                    if not c.isupper():
                        if c == parent:
                            args.append(np.einsum(head+'->'+char[:len(rule['parent'])], rule['param']))
                        else:
                            if c == 'root':
                                args.append(rule['param'])
                            else:
                                args.append(inside_sweep(c, param))
                #print(args)
                return np.einsum(*args)


def compare_rules(rule1, rule2):
    if rule1['parent'] == rule2['parent']:
        if rule1['children'] == rule2['children']:
            return True
    return False

def list_eq(a,b):
    c = copy.deepcopy(a)
    d = copy.deepcopy(b)
    #make a deep copy or the original order is messed up, cuz sorted() is inplace
    return sorted(c) == sorted(d)

def rule_eq(p1, p2, c1, c2):
    return list_eq(p1,p2) and list_eq(c1,c2)

def param_index(parent, children, param):
    for rule in param:
        if rule_eq(parent, rule['parent'], children, rule['children']):
            return rule
    return []

def get_rules_from_tree(node):
    batch = [node]
    rules = []
    while batch != []:
        lv = []
        for item in batch:
            if not(item.type.isupper()):
                lv = lv + item.children
                rule = {'parent':[item.type], 'children':sorted([c.type for c in item.children])}
                rules.append(rule)
        batch = lv
    return rules

def agenda_search(obs_facts, param):
    tmp = copy.deepcopy(obs_facts)
    obs_types = [x.type for x in obs_facts]
    obs = collections.Counter(obs_types)
    batch = []
    batch_new = []
    parents = []
    flag = True
    last_change = []
    while flag:
        hist = []
        for rule in param:
            tmp_rule = collections.Counter(rule['children'])
            rule_flag = []
            for key in tmp_rule.keys():
                if key in obs.keys() and obs[key] - tmp_rule[key] >=0:
                    rule_flag.append(True)
                else:
                    rule_flag.append(False)
            if all(rule_flag):
                batch_new.append(rule)
                parent = Tree(rule['parent'][0])
                for key in tmp_rule.keys():
                    obs[key] = obs[key] - tmp_rule[key]
                for i in rule['children']:
                    for node in tmp:
                        if node.type == i:
                            parent.add_child(node)
                            node.add_parent(parent)
                            tmp.remove(node)
                            break
                parents.append(parent)
                hist.append(parent)
        last_change.append(hist)
        if len(batch_new) == len(batch):
            flag = False
            for p in last_change[-1]:
                p.set_root()
        else:
            batch = batch_new

    return batch, parents
