from treeType import *
from utils import *
from evaluator import *
from copy import deepcopy
import torch
import numpy as np
import collections
import time
import pickle
import sys
import json

class Pgd(object):

    def __init__(self, data, test, rules, latent, max_itr, learning_rate):
        #use log parameters for unconstrained optimization
        # v = log(sigmoid(y))
        # sigmoid(y) = exp(v)
        self.data = data
        self.test = test
        self.rules = rules
        self.latent = latent
        self.parameters = [] #parameters are in log
        self.lr = learning_rate
        self.mode = 'count'

        self.root_parameters = [] #parameters are in log 
        self.logsoftmax = torch.nn.LogSoftmax()
        for rule in rules:
            if all([x.isupper() for x in rule['children']]):
                param = np.random.random(self.latent)
            else:
                size = len([rule['parent']]) + len(rule['children'])
                param = np.random.random(self.latent)
                
                for i in range(size - 1):
                    param = np.multiply.outer(param, np.random.random(self.latent))

            param = torch.from_numpy(param)
            param.requires_grad = True
            current = {'parent':[rule['parent']], 'children':rule['children'], 'param': param, 'count': 0}
            self.parameters.append(current)
            tmp = torch.from_numpy(np.random.random(self.latent))
            tmp.requires_grad = True
            root = {'parent':[rule['parent']], 'children':['root'], 'param': tmp, 'count':0}
            self.root_parameters.append(root)
    

    def forward(self, tree):
        batch = [tree]
        nll = 0
        while batch != []:
            lv = []
            for x in batch:
                obs_not_flag = []
                for c in x.children:
                    if c.type.isupper():
                        obs_not_flag.append(False)
                    else:
                        obs_not_flag.append(True)
                if all(obs_not_flag):
                    lv = lv + x.children

            for p in batch:
                nll = nll - torch.dot(self.inside_sweep(p.type), self.outside_sweep(p.type))
            
            batch = lv
        return nll


    def inside_sweep(self, parent):
        parent_type = [parent]
        char = 'abcdefghijklmno'
        for rule in self.parameters:
            if parent in rule['parent']:
                if all([x.isupper() for x in rule['children']]):
                    return self.logsoftmax(rule['param'])
                else:
                    children_type = rule['children']
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
                                args.append(self.inside_sweep(c))
                        else:
                            args.append(torch.einsum(head+'->'+char[:len(parent_type)], rule['param']))
                    #print(args)
                    #print(parent)
                    #print(torch.einsum(*args))
                    return self.logsoftmax(torch.einsum(*args))
    
    def outside_sweep(self, parent):
        char = 'abcdefghijklmno'
        for rule in self.root_parameters:
            if rule['parent'] == [parent] and rule['children'] == ['root']:
                return self.logsoftmax(rule['param'])
        else:
            for rule in self.parameters:
                if parent in rule['children']:
                    outside = self.outside_sweep(rule['parent'])
                    formula = char[:len(rule['parent'])] + char[len(rule['parent']): len(rule['parent']) + len(rule['children'])]
                    head = formula
                    for l in range(len(rule['children'])):
                        formula = formula + ',' + char[len(rule['parent']) + l] # shape of individual vector
                    formula = formula + '->' + char[len(rule['parent']) + rule['children'].index(parent)]
                    
                    #print(formula)
                    args = [formula, self.logsoftmax(rule['param'])]
                    for c in rule['children']:
                        #print(c)
                        if not c.isupper():
                            if c == parent:
                                args.append(torch.einsum(head+'->'+char[:len(rule['parent'])], self.logsoftmax(['param'])))
                            else:
                                if c == 'root':
                                    args.append(self.logsoftmax(rule['param']))
                                else:
                                    args.append(self.inside_sweep(c))
                    #print(args)
                    return torch.einsum(*args)

    def approximate_search(self, types, method):
        items_dict = collections.Counter(types)
        ans = []
        rule_used = []

        if method == 'prior':
            tmp_param = sorted(self.parameters, key= lambda x: x['count'], reverse=True)
        elif method == 'bag':
            tmp_param = sorted(self.parameters, key= lambda x: len(x['children']), reverse=True)
        else:
            tmp_param = deepcopy(self.parameters)
            random.shuffle(tmp_param)

        no_change = True
        while no_change:
            no_change = False
            for rule in tmp_param:
                tmp_counter = collections.Counter(rule['children'])
                
                change_flags = []
                for i in tmp_counter.keys():
                    if i in items_dict.keys() and (items_dict[i] - tmp_counter[i]) >= 0:
                        change_flags.append(True)
                    else:
                        change_flags.append(False)
                
                if all(change_flags):
                    ans = ans + rule['parent']
                    rule_used.append(rule)
                    for i in tmp_counter.keys():
                        items_dict[i] = items_dict[i] - tmp_counter[i]
                    
                    no_change = True
            
        return ans, rule_used

    def train(self):
            avg_nll = 0
            start_time = time.time()
            for tree in self.data[:100]:
                nll = self.forward(tree)
                #print(nll)
                nll.backward()
                avg_nll += nll
                for rule in self.parameters:
                    #try:
                    if rule['param'].grad != None:
                        if (rule['param'].grad.sum() !=0):
                            with torch.no_grad():
                                rule['param'] -= pgd.lr * torch.clamp(rule['param'].grad, min = -1, max =1)
                                #print(rule['param'])
                                #tmp_grad = rule['param'] - self.lr * rule['param'].grad
                                #perform projected gradient descent
                                #print(rule['param'].grad)
                                #rule['param'] = tmp_grad
                            rule['count'] += 1
                            rule['param'].grad.zero_()
                        #print(rule['param'].grad)
                    #except:pass
            avg_nll = avg_nll / len(self.data)
            print('Time Elapsed:{}'.format(time.time() - start_time))
            #print('Avg NLL:{}'.format(avg_nll))
    
    def predict(self, obs):
        ans = []
        batch = [x.type for x in obs]
        lv_count = 0
        
        while batch !=[] and lv_count <5:
            a, _ = self.approximate_search(batch, self.mode)
            ans = ans + a
            batch = a
            lv_count +=1
        
        return ans

    def macro_avg(self):
        precision = 0
        recall = 0
        for i in tqdm(range(len(self.test)), desc = 'tqdm() Progress Bar'):
            tree = self.test[i]
            golden = all_nodes(tree)
            parsed_types = self.predict(get_obs_var(tree))
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
        
        return precision/len(self.test), recall/len(self.test)

    def micro_avg(self):
        precision = 0
        recall = 0
        parsed_types = []
        golden_types = []
        for i in tqdm(range(len(self.test)), desc = 'tqdm() Progress Bar'):
            tree = self.test[i]
            golden = all_nodes(tree)
            parsed_types = parsed_types + self.predict(get_obs_var(tree))
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

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    results = {}
    for i in range(5):
        with open('data-{}.pkl'.format(i), 'rb') as f:
            data = pickle.load(f)

        pgd = Pgd(data['train'], data['test'], data['program'].rules, 5, 1000, 0.1)
        print('data-{}'.format(i))
        for j in range(1):
            pgd.train()
        
        result = []
        result.append(pgd.micro_avg())
        result.append(pgd.macro_avg())
        pgd.mode = 'bag'
        result.append(pgd.micro_avg())
        result.append(pgd.macro_avg())
        results[i] = result
    with open('out_pgd.txt', 'w') as f:
        json.dump(results, f)