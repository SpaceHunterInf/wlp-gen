from treeType import *
import numpy as np


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
    for rule in relations:
        if parent in rule['parent']:
            if all([x.isupper() for x in rule['children']]):
                return rule['param']
            else:
                children_type = rule['children']
                transition = rule['param']
                formula = char[:len(parent_type)] + char[len(parent_type):len(parent_type) + len(children_type)] #shape of transiton
                for l in range(len(children_type)):
                    formula = formula + ',' + char[len(parent_type) + l] # shape of individual vector
                formula = formula + '->' +  char[:len(parent_type)]
                print(formula)
                
                args = [formula, rule['param']]
                for c in rule['children']:
                    args.append(inside_sweep(c, param))
                return np.einsum(*args)

def outside_sweep(parent, param):
    char = 'abcdefghijklmno'
    for rule in param:
        if rule['parent'] == node.type and rule['children'] == ['root']:
            return rule['param']
    else:
        for rule in param:
            if parent in rule['children']:
                siblings = [x for x in rule['children'] if x != node.type]
                outside = outside_sweep(rule['parent'], param)
                
                formula = char[:len(rule['parent'])] + char[len(rule['parent']): len(rule['parent']) + len(rule['children'])]
                for l in range(len(siblings)):
                    formula = formula + ',' + char[len(rule['parent']) + l] # shape of individual vector
                formula = formula + '->' + char[len(rule['parent']) + rule['children'].index(parent)]
                
                args = [formula, rule['param']]
                for c in siblings:
                    args.append(inside_sweep(c, param))
                return np.einsum(*args)


def compare_rules(rule1, rule2):
    if rule1['parent'] == rule2['parent']:
        if rule1['children'] == rule2['children']:
            return True
    return False
    