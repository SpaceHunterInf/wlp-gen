import queue
from secrets import randbelow
import numpy as np
import random
from treeType import *
from utils import *
from tqdm import tqdm
import time
import pydot
import pickle

random.seed(2022)
np.random.seed(2022)


np.set_printoptions(threshold=np.inf)

def preterminal_match(pres, obs, vars, max_co, latent_table):
    pre_obs_relations = []
    var_pre_relations = []
    for pre in pres:
        children = []
        latent = latent_table[pre]
        for j in random.sample(range(len(obs)), latent):
            children.append(obs[j])
        dist = np.random.dirichlet(np.ones(latent))
        children = sorted(children)
        rule = {'parent':pre, 'children':children, 'dist': dist}

        pre_obs_relations.append(rule)

    for var in vars:
        latent = latent_table[var]
        children = np.random.choice(pres, np.random.randint(1,max_co))
        parent_dis = np.random.dirichlet(np.ones(latent))
        children = sorted(children)
        for child in children:
            child_dis = np.random.dirichlet(np.ones(latent_table[child]))
            parent_dis = np.multiply.outer(parent_dis, child_dis)
        
        rule = {'parent':var, 'children':children, 'dist': parent_dis}
        var_pre_relations.append(rule)
    
    return pre_obs_relations, var_pre_relations

def variable_match(variable, max_co, latent_table):
    relations = []
    for var in variable:
        parent_latent = latent_table[var]
        chilren_num = np.random.randint(1,max_co)
        children = np.random.choice(variable, chilren_num) # using sample with replacement to allow recursiveness
        children = sorted(children)
        parent_dis = np.random.dirichlet(np.ones(parent_latent))
        for child in children:
            child_dis = np.random.dirichlet(np.ones(latent_table[child]))
            parent_dis = np.multiply.outer(parent_dis, child_dis)
        
        rule = {'parent':var, 'children':children, 'dist': parent_dis}
        relations.append(rule)

    #construct simple rules

    multi_proof = np.random.choice(relations, 5)
    for rule in multi_proof:
        var = rule['parent']
        parent_latent = latent_table[var]
        children = np.random.choice(variable, chilren_num) # using sample with replacement to allow recursiveness
        children = sorted(children)
        parent_dis = np.random.dirichlet(np.ones(parent_latent))
        for child in children:
            child_dis = np.random.dirichlet(np.ones(latent_table[child]))
            parent_dis = np.multiply.outer(parent_dis, child_dis)

        rule = {'parent':var, 'children':children, 'dist': parent_dis}
        relations.append(rule)
    #add multiple proof to 1 variable

    multi_proof = np.random.choice(relations, 5)
    for rule in multi_proof:
        children = rule['children']
        children = sorted(children)
        var = np.random.choice(variable, 1)[0]
        parent_latent = latent_table[var]

        parent_dis = np.random.dirichlet(np.ones(parent_latent))
        for child in children:
            child_dis = np.random.dirichlet(np.ones(latent_table[child]))
            parent_dis = np.multiply.outer(parent_dis, child_dis)
        
        rule = {'parent':var, 'children':children, 'dist': parent_dis}
        relations.append(rule)
    #add multiple parent to same group of children

    clean = []
    for i in range(len(relations)):
        for j in range(len(relations)):
            if relations[i]['parent'] == relations[j]['parent']:
                flag1 = True
                flag2 = True
                for x in relations[i]['children']:
                    if not (x in relations[j]['children']):
                        flag1 = False
                for x in relations[j]['children']:
                    if not (x in relations[i]['children']):
                            flag2 = False
                if (flag1 == True) and (flag2 == True):
                    if i != j:
                        relations[j]['del'] = True

    for i in relations:
        if not ('del' in i.keys()):
            clean.append(i)
            print(str(i['parent']) + ' --> ' + str(i['children']))
    #remove all duplicates
    return clean        

class singleGen:
    def __init__(self, max_latent, max_axiom, max_correlation, max_depth):
        self.max_latent = max_latent
        self.max_depth = max_depth
        self.max_axiom = max_axiom
        self.max_correlation = max_correlation
        self.axiom_chr = 'abcdefghijklmnopqrstuvwxyz'
        self.observed_chr = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.obs_len = 3
        self.obs_amount = 40
        self.axi_len = 5
        self.preterminal_prop = 20
        #setting up hyperparameters
        self.obs_facts = []
        self.preterminal = []
        self.variables = []
        self.latent_table = {} #storing variable-number of latent state


        for i in range(self.obs_amount):
            obs = ''
            for j in range(np.random.randint(1, self.obs_len)):
                obs = obs + self.observed_chr[np.random.randint(25)]
            self.obs_facts.append(obs)
        #creating conditional input

        for i in range(self.preterminal_prop):
            axiom = ''
            for j in range(np.random.randint(1, self.axi_len)):
                axiom = axiom + self.axiom_chr[np.random.randint(25)]
            self.preterminal.append(axiom)
        #creating preterminals

        for i in range(self.max_axiom):
            axiom = ''
            for j in range(np.random.randint(1, self.axi_len)):
                axiom = axiom + self.axiom_chr[np.random.randint(25)]
            if not(axiom in self.preterminal):
                self.variables.append(axiom)

        for term in self.preterminal + self.variables:
            self.latent_table[term] = np.random.randint(1,max_latent)

        #self.preterminal_rules = preterminal_match(self.preterminal, self.obs_facts, self.max_correlation, self.max_latent)
        self.variables_rules = variable_match(self.variables,self.max_correlation, self.latent_table)
        self.prior_goal_dist = np.random.dirichlet(np.ones(len(self.variables)))

        self.same_parent_split = {}
        for var in self.variables:
            self.same_parent_split[var] = {'children':[], 'dist':None}
            for rule in self.variables_rules:
                if rule['parent'] == var:
                    self.same_parent_split[var]['children'].append(rule['children'])
            self.same_parent_split[var]['dist'] = np.random.dirichlet(np.ones(len(self.same_parent_split[var]['children'])))
        #initialize rules and each variable to be the goal proof

        self.pre_obs_relations, self.var_pre_relations = preterminal_match(self.preterminal, self.obs_facts, self.variables, self.max_correlation, self.latent_table)
        self.same_preterminal_split = {}
        for var in self.preterminal + self.variables:
            self.same_preterminal_split[var] = {'children':[], 'dist':None}
            for rule in self.pre_obs_relations + self.var_pre_relations:
                if rule['parent'] == var:
                    self.same_preterminal_split[var]['children'].append(rule['children'])
            self.same_preterminal_split[var]['dist'] = np.random.dirichlet(np.ones(len(self.same_preterminal_split[var]['children'])))

        self.rules = self.pre_obs_relations + self.var_pre_relations + self.variables_rules
        self.types = self.preterminal + self.variables

    def get_proof(self):
        goal = Tree(np.random.choice(self.variables, 1, p=self.prior_goal_dist)[0])
        goal.set_root()
        batch = [goal]
        for i in range(self.max_depth):
            #print(i)
            next_lvl = []
            for var in batch:
                var_type = var.type
                #var.set_suffix(str(i))
                self.same_parent_split[var_type]['children']
                #print(self.same_parent_split[var]['dist'])
                children_idx = np.random.choice(len(self.same_parent_split[var_type]['children']), 1, p=self.same_parent_split[var_type]['dist'])
                #print(children_idx[0])
                children = self.same_parent_split[var_type]['children'][children_idx[0]]
                children = sorted(children)
                #print(children)

                childrenset = set(children)
                for type in childrenset:
                    idx = 0
                    for child in children:
                        if type == child:
                            subtree = Tree(type)
                            subtree.set_prefix(str(idx))
                            subtree.set_suffix(str(i))
                            var.add_child(subtree)
                            next_lvl.append(subtree)
                            #print(subtree.name)
                            idx = idx + 1
                #Assinging different idx to same type variable within 1 rule eg: a --> b b b ==> a --> 0-b 1-b 2-b
            batch = next_lvl
            #print(len(batch))
            # except:
            #     print("shit happens")
            #     pass
        #TODO debug tomorrow
        return goal
    
    def get_full_tree(self, goal):
        batch = [goal]
        while batch != []:
            lv = []
            for node in batch:
                if node.children != []:
                    lv = lv + node.children 
                else:
                    var_type = node.type
                    children_idx = np.random.choice(len(self.same_preterminal_split[var_type]['children']), 1, p=self.same_preterminal_split[var_type]['dist'])
                    #print(children_idx)
                    #print(self.same_preterminal_split[var_type]['children'][children_idx[0]])
                    children = self.same_preterminal_split[var_type]['children'][children_idx[0]]
                    children = sorted(children)
                    childrenset = set(children)
                    #print(childrenset)
                    for type in childrenset:
                        idx = 0
                        for child in children:
                            #print(child in self.preterminal)
                            if type == child:
                                subtree = Tree(type)
                                subtree.set_prefix(str(idx))
                                node.add_child(subtree)
                                idx = idx + 1
                    for child in node.children:
                        var_type = child.type
                        #print(var_type in self.preterminal)
                        children_idx = np.random.choice(len(self.same_preterminal_split[var_type]['children']), 1, p=self.same_preterminal_split[var_type]['dist'])
                        children = self.same_preterminal_split[var_type]['children'][children_idx[0]]
                        children = sorted(children)
                        childrenset = set(children)
                        #print(childrenset)
                        #print('test')
                        for type in childrenset:
                            idx = 0
                            for grand_child in children:
                                #print(grand_child in self.obs_facts)
                                if type == grand_child:
                                    subtree = Tree(type)
                                    subtree.set_prefix(str(idx))
                                    child.add_child(subtree)
                                    idx = idx + 1
            batch = lv
        return goal


def print_rule(relations):

    for i in relations:
        print(str(i['parent']) + ' --> ' + str(i['children']))

def batch_generation():
    test = singleGen(5,20,5,3)
    data = {'train':[], 'dev':[], 'test':[], 'program': test}
    
    total = 1000
    for i in tqdm(range(total), desc = 'tqdm() Progress Bar'):
        a = test.get_proof()
        b = test.get_full_tree(a)
        
        if i <0.8*total:
            data['train'].append(b)
        elif i>0.8*total and i<0.9*total:
            data['dev'].append(b)
        else:
            data['test'].append(b)
    
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)



test = singleGen(5,20,5,3)
#print(test.obs_facts)
a = test.get_proof()
#print_rule(test.pre_obs_relations)
#print_rule(test.var_pre_relations)
b = test.get_full_tree(a)
bec = b.children[0].get_inside()
vec = b.children[0].get_outside()
print(bec)
print(vec)
#print(feature_gen(bec, test.obs_facts + test.preterminal + test.variables, 10000))
print(len(test.obs_facts + test.preterminal + test.variables))
batch_generation()