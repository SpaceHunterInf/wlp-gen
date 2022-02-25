from copy import deepcopy
from genericpath import exists
from this import d
from treeType import *
import numpy as np
import collections
import numpy as np
from tqdm import tqdm
from utils import *
import random

def dfs_parse(Items : list, Rules : list):
    ans = []
    Items = collections.Counter(Items)
    now_counter = collections.Counter()
    def check():
        for k, v in now_counter.items():
            if(Items[k] < v):
                return False
        return True
    now_choose = []
    def dfs(now):
        if (now > 10000):
            return
        nonlocal now_choose, now_counter
        if(now == len(Rules)):
            if(now_counter == Items):
                ans.append(now_choose[:])
            return
        cnt = 0
        tmp = collections.Counter(Rules[now])
        while(True):
            now_counter += tmp
            cnt += 1
            now_choose.append(Rules[now])
            if(check()):
                dfs(now + 1)
            else:
                break
        for i in range(cnt):
            now_counter -= tmp
            now_choose.pop()

        dfs(now + 1)
    dfs(0)
    return ans

def parse_encap(nodes, relations):
    items = [x.type for x in nodes]
    rules = [x['children'] for x in relations]
    ans_children = dfs_parse(items, rules)
    
    ans_parents = []
    for i in ans_children:
        ans_buffer = []
        for c in i:
            ans_tmp = []
            for rule in relations:
                if sorted(rule['children']) == sorted(c):
                    parent_node = Tree(rule['parent'])
                    ans_tmp.append(parent_node)
                    break
            ans_buffer.append(ans_tmp)
        assert(len(ans_buffer) == len(i))
        ans_parents.append(ans_buffer)
    assert(len(ans_parents) == len(ans_children))
    return ans_children, ans_parents
    #return ans_parents

def silly_parse(node, pre_obs, var_rules, parameters):
    
    obs_groups, pre_groups = parse_encap(get_obs_var(node), pre_obs)
    pre_terminals = [{'queue': x, 'buffer':x} for x in pre_groups]
    
    #print(pre_terminals)
    i = 0
    batch = pre_terminals
    batch_buffer = pre_terminals
    while i < 5 and len(batch_buffer) != 0 :
        if len(batch) > 2:
            batch = batch[:2]

        batch_buffer = []
        for proof in batch:
            in_feed = []
            for group in proof['queue']:
                in_feed = in_feed + group
            #print(in_feed)
            ans_children, ans_parents = parse_encap(in_feed, var_rules)
            for parents in ans_parents:
                next_proof_lv = parents
                buffer = proof['buffer'] + parents
                batch_buffer.append({'queue': next_proof_lv, 'buffer':buffer})
        
        #print(batch_buffer)
        if len(batch_buffer) != 0:
            batch = batch_buffer
        i = i + 1
    
    return batch

def all_nodes(tree):
    ans = []
    batch = [tree]
    while batch != []:
        lv = []
        for node in batch:
            if not(node.type.isupper()):
                ans.append(node)
            lv = lv + node.children
        batch = lv
    return ans

def compare_node_types(nodes, types):
    tmp_nodes = deepcopy(nodes)
    tmp_types = deepcopy(types)
    prev_len = len(tmp_types)
    unchanged = False
    while unchanged == False:
        for node in tmp_nodes:
            for t in tmp_types:
                if node.type == t:
                    tmp_nodes.remove(node)
                    tmp_types.remove(t)
                    break
        if prev_len == len(tmp_types):
            unchanged = True
        else:
            prev_len = len(tmp_types)
    return tmp_nodes, tmp_types

def get_obs_var(tree):
        obs = []
        var = []
        batch = [tree]
        while batch != []:
            lv = []
            for node in batch:
                lv = lv + node.children
                if node.type.isupper():
                    obs.append(node)
                else:
                    var.append(node)
            batch = lv
        return obs


def get_best_parse(obs, parameters, root_parameters):
    current_parents = obs
    current_types = [x.type for x in obs]
    res = None
    ans = []
    for i in range(5): # iterate through 5 levels at most
        ranking = []
        for j in range(5): # beam search width = 5
            types, parents = agenda_search(current_parents, parameters)
            score = -100
            for x in parents:
                t = x.type
                try:
                    if x.root:
                        score = score + root_parameters[t].dot(inside_sweep(t, parameters))
                    else:
                        score = score + np.log(inside_sweep(t, parameters).dot(outside_sweep(t, parameters+root_parameters)))
                except:
                    pass
            if len(parents) > 0:
                ranking.append({'parent':parents, 'score':score})
        ranking = sorted(ranking, key= lambda x: x['score'], reverse=True)
        if len(ranking) > 0:
            current_parents = ranking[0]['parent']
            ans = ans + [p.type for p in current_parents]
    
    return ans

class evaluator(object):
    def __init__(self, parameters, root_parameters, data):
        self.parameters = parameters
        self.root_parameters = root_parameters
        self.data = data
        self.rules = None

    def macro_avg(self):
        precision = 0
        recall = 0
        for i in tqdm(range(len(self.data)), desc = 'tqdm() Progress Bar'):
            tree = self.data[i]
            golden = all_nodes(tree)
            parsed_types = get_best_parse(get_obs_var(tree), self.parameters, self.root_parameters)
            golden_types = [x.type for x in golden]

            parse_count = collections.Counter(parsed_types)
            golden_count = collections.Counter(golden_types)

            correct_count = {}
            total_correct = 0

            for t in golden_count.keys():
                if t in parse_count:
                    if parse_count[t] <= golden_count[t]:
                        correct_count[t] = parse_count[t]
                        total_correct = total_correct + parse_count[t]
                    else:
                        correct_count[t] = golden_count[t]
                        total_correct = total_correct + golden_count[t]

            pre = total_correct/len(parsed_types)
            rec = total_correct/len(golden_types)
            precision = precision + pre
            recall = recall + rec
        
        return precision/len(self.data), recall/len(self.data)

    def micro_avg(self):
        precision = 0
        recall = 0
        parsed_types = []
        golden_types = []
        for i in tqdm(range(len(self.data)), desc = 'tqdm() Progress Bar'):
            tree = self.data[i]
            golden = all_nodes(tree)
            parsed_types = parsed_types + get_best_parse(get_obs_var(tree), self.parameters, self.root_parameters)
            golden_types = golden_types + [x.type for x in golden]

        parse_count = collections.Counter(parsed_types)
        golden_count = collections.Counter(golden_types)

        correct_count = {}
        total_correct = 0

        for t in golden_count.keys():
            if t in parse_count:
                if parse_count[t] <= golden_count[t]:
                    correct_count[t] = parse_count[t]
                    total_correct = total_correct + parse_count[t]
                else:
                    correct_count[t] = golden_count[t]
                    total_correct = total_correct + golden_count[t]

        pre = total_correct/len(parsed_types)
        rec = total_correct/len(golden_types)
        precision = precision + pre
        recall = recall + rec
        
        return precision, recall
        

    
print(dfs_parse([1, 2, 3], [[1, 2], [1], [2], [3], [1,2,3,4,5], [1,3]]))
        
                            
                        


    

    