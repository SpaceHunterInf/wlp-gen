import queue
import numpy as np
import random
from treeType import *
random.seed(2022)
np.random.seed(2022)
import pydot

def preterminal_match(pres, obs, max_co, max_latent):
    relations = []
    for pre in pres:
        children = []
        latent = np.random.randint(1, max_co)
        for j in random.sample(range(len(obs)), latent):
            children.append(obs[j])
        dist = np.random.dirichlet(np.ones(np.random.randint(1,max_latent)))

        rule = {'parent':pre, 'children':children, 'dist': dist}

        relations.append(rule)
    return relations        

def variable_match(variable, max_co, latent_table):
    relations = []
    for var in variable:
        parent_latent = latent_table[var]
        chilren_num = np.random.randint(1,max_co)
        children = np.random.choice(variable, chilren_num) # using sample with replacement to allow recursiveness

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
        self.obs_amount = 1000
        self.axi_len = 3
        self.preterminal_prop = int(0.1 * self.max_axiom)
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

    def get_proof(self):
        goal = Tree(np.random.choice(self.variables, 1, p=self.prior_goal_dist)[0])
        batch = [goal]
        for i in range(self.max_depth):
            print(i)
            next_lvl = []
            for var in batch:
                var_type = var.type
                #var.set_suffix(str(i))
                self.same_parent_split[var_type]['children']
                #print(self.same_parent_split[var]['dist'])
                children_idx = np.random.choice(len(self.same_parent_split[var_type]['children']), 1, p=self.same_parent_split[var_type]['dist'])
                #print(children_idx[0])
                children = self.same_parent_split[var_type]['children'][children_idx[0]].tolist()
                print(children)

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
                            print(subtree.name)
                            idx = idx + 1
                #Assinging different idx to same type variable within 1 rule eg: a --> b b b ==> a --> 0-b 1-b 2-b
            batch = next_lvl
            print(len(batch))
            # except:
            #     print("shit happens")
            #     pass

        #TODO debug tomorrow
        return goal


test = singleGen(5,20,5,3)
#print(test.obs_facts)
a = test.get_proof()
draw_tree(a)