from inspect import Parameter
from locale import normalize
import numpy as np
from evaluator import *
from utils import *
from treeType import *
import math
import random
import pickle,sys,json
from gen import *

np.random.seed(2022)

class EM(object):
    # EM algorithm for estimating L-WLP
    def __init__(self, max_iteration, data, rules, latent, program):
        self.max_it = max_iteration
        self.data = [get_obs_var(tree) for tree in data]
        self.latent = latent
        self.expected = []
        self.types = program.types
        self.root_parameters = []

        self.parameters = []
        for rule in rules:
            if all([x.isupper() for x in rule['children']]):
                param = np.random.dirichlet(np.ones(self.latent))
            else:
                size = len([rule['parent']]) + len(rule['children'])
                param = np.random.dirichlet(np.ones(self.latent))
                
                for i in range(size - 1):
                    param = np.multiply.outer(param, np.random.dirichlet(np.ones(self.latent)))

            current = {'parent':[rule['parent']], 'children':rule['children'], 'param': param, 'count': np.random.randint(0, len(rules))}
            self.parameters.append(current)
        # randomly inistialize all rules with a fixed size of latent state probabilities
        self.inside_outside = {}

        for i in program.types:
            if not(i.isupper()):
                self.inside_outside[i] = {'inside': np.random.dirichlet(np.ones(self.latent)), 'outside': np.random.dirichlet(np.ones(self.latent))}
                self.root_parameters.append({'parent':[i], 'children':['root'], 'param':np.random.dirichlet(np.ones(self.latent))})
    
    def parameter_update(self):
        new_params = []
        ll = 0
        count = 0
        for tree_list in self.expected: #indexed per sentence
            for tree in tree_list:
                g = tree.type
                g = self.inside_outside[g]['inside'].dot(param_index([g], ['root'], self.parameters+self.root_parameters)['param'])
                
                if g!=0:
                    print(g)
                    ll += math.log(g)
                    print("Log Likelihood: %.3f"%ll)
                rules = get_rules_from_tree(tree)
                count += 1
                for rule in rules: #for each minimal rule for the sentence, have inside/outside probs of spans
                    result = self.inside_outside[rule['parent'][0]]['outside']
                    if not all([c.isupper() for c in rule['children']]):
                        for c in rule['children']:
                            result = np.multiply.outer(result, self.inside_outside[c]['inside'])
                    history = param_index(rule['parent'], rule['children'], new_params)
                    if history == []:
                        new_params.append({'parent':rule['parent'], 'children':rule['children'], 'param': result, 'count': 1})
                    else:
                        history['param'] = history['param'] + result
                        history['count'] = history['count'] + 1
        
        for rule in new_params:
            rule['param'] = rule['param'] / rule['count'] #normalize
        return new_params

    def expectation_step(self):
        expected = []
        for obs in self.data[:10]:
            current_parents = obs
            current_types = [x.type for x in obs]
            res = None
            for i in range(5):
                ranking = []
                for j in range(5):
                    random.shuffle(self.parameters)
                    types, parents = agenda_search(current_parents, self.parameters)
                    score = -100
                    for x in parents:
                        t = x.type
                        try:
                            if x.root:
                                score = score + self.root_parameters[t].dot(inside_sweep(t, self.parameters))
                            else:
                                score = score + np.log(inside_sweep(t, self.parameters).dot(outside_sweep(t, self.parameters+self.root_parameters)))
                        except:
                            pass
                    if len(parents) > 0:
                        ranking.append({'parent':parents, 'score':score})
                ranking = sorted(ranking, key= lambda x: x['score'], reverse=True)
                #print(ranking)
                if len(ranking) > 0:
                    res = ranking[0]['parent']
                #current_types = ranking[0]['type']
                    current_parents = res
            #print(res)
            expected.append(res)
        self.expected = expected

    def maximization_step(self):
        in_out = {}
        for i in self.types:
            in_out[i] = {'inside':np.zeros(self.latent), 'outside':np.zeros(self.latent), 'count':0}
        for tree_list in self.expected:
            for tree in tree_list:
                types = [x.type for x in all_nodes(tree)]
                for t in types:
                    try:
                        in_out[t]['inside'] = in_out[t]['inside'] + inside_sweep(t)
                        in_out[t]['outside'] = in_out[t]['outside'] + outside_sweep(t)
                    except:
                        pass
                    in_out[t]['count']  = in_out[t]['count'] + 1

        normalized_inout = {}
        for i in self.types:
            if in_out[i]['count'] != 0:
                normalized_inout[i] = {'inside':in_out[i]['inside'] / in_out[i]['count'], 'outside':in_out[i]['outside'] / in_out[i]['count']}
        
        self.inside_outside = normalized_inout
        self.parameters = self.parameter_update()

        new_root_param = []
        for i in self.types:
            new_root_param.append({'parent':[i], 'children':['root'], 'param':np.zeros(self.latent), 'count':0})
        for tree_list in self.expected:
            for tree in tree_list:
                if tree.root:
                    tmp = param_index([tree.type], ['root'], new_root_param)
                    tmp['count']  = tmp['count']+ 1
                    try:
                        tmp['param'] = inside_sweep(tree.type, self.parameters)
                    except:
                        pass
        
        for i in self.types:
            tmp = param_index([i], ['root'], new_root_param)
            if tmp['count'] != 0:
                tmp['param'] = tmp['param'] / tmp['count']
        
        self.root_parameters = new_root_param


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    results = {}
    for i in range(5):
        with open('data-{}.pkl'.format(i),'rb') as f:
            data = pickle.load(f)
        model = EM(1000, data['train'], data['program'].rules, 5, data['program'])

        for j in range(1000):
            model.expectation_step()
            model.maximization_step()
        parameters = model.parameters
        root_parameters = model.root_parameters

        test = evaluator(parameters, root_parameters, data['test'], get_best_parse)
        print('data-{}'.format(i))
        results[i] = [test.micro_avg(), test.macro_avg()]

    with open('out_em.txt', 'w') as f:
        json.dump(results, f)
