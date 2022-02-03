import numpy as np

class EM(object):
    # EM algorithm for estimating L-WLP
    def __init__(self, max_iteration, data, rules, latent):
        self.max_it = max_iteration
        self.data = data
        self.latent = latent

        self.parameters = []
        for rule in rules:
            size = len(rule['parent']) + len(rule['children'])
            param = np.random.dirichlet(np.ones(self.latent))
            
            for i in range(size - 1):
                param = np.outer(param, np.random.dirichlet(np.ones(self.latent)))
            current = {'parent':rule['parent'], 'children':rule['children'], 'param': param}
            self.parameters.append(current)
        # randomly inistialize all rules with a fixed size of latent state probabilities

    
    def inside_prob(self, obs_facts):
        pass
