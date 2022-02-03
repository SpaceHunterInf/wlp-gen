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
    

def inside_prob(node, relations, latent):
    m = latent #number of latent states
    
    children = node.children
    current_rule = None

    for rule in relations:
        if rule['parent'] == node.type:
            flag = True
            for child in children:
                if child.type in rule['children']:
                    flag = False
            if flag == True:
                current_rule = rule

    if current_rule == None:
        print('Rule not found')
        return np.zeros(m)

def compare_rules(rule1, rule2):
    if rule1['parent'] == rule2['parent']:
        if rule1['children'] == rule2['children']:
            return True
    return False
    