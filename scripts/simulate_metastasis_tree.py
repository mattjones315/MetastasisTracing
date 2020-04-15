import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as scs
import networkx as nx

from collections import defaultdict

import subprocess
import os

from cassiopeia.TreeSolver.simulation_tools import dataset_generation as data_gen

import utilities.metastasis_score_utils as met_utils

def compute_priors(C, S, p, mean=0.01, disp=0.1, empirical = np.array([])):
    
    sp = {}
    prior_probabilities = {}
    for i in range(0, C):
        if len(empirical) > 0:
            sampled_probabilities = sorted(empirical)
        else:
            sampled_probabilities = sorted([np.random.negative_binomial(mean,disp) for _ in range(1,S+1)])
        mut_rate = p
        prior_probabilities[i] = {'0': (1-mut_rate)}
        total = np.sum(sampled_probabilities)

        sampled_probabilities = list(map(lambda x: x / (1.0 * total), sampled_probabilities))
            
        for j in range(1, S+1):
            prior_probabilities[i][str(j)] = (mut_rate)*sampled_probabilities[j-1]

    return prior_probabilities, sp 

def get_transition_stats(tree):
    
    n_transitions = 0
    transitions = defaultdict(dict)
    freqs = defaultdict(int)
    
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    for e in nx.dfs_edges(tree, source=root):
        
        p,c = e[0], e[1]
        m_p, m_c = tree.nodes[p]['meta'], tree.nodes[c]['meta']
        if m_p != m_c:
            n_transitions += 1
            if m_c not in transitions[m_p]:
                transitions[m_p][m_c] = 0
            transitions[m_p][m_c] += 1
            
        if tree.out_degree(c) == 0:
            freqs[m_c] += 1
            
    return n_transitions, transitions, freqs

no_mut_rate = 0.985
number_of_states = 40
dropout = 0.17
depth = 11
number_of_characters = 40

N_clones = 900 #number of clones to simulate
max_mu = 0.3 #max rate of metastasis
min_alpha = 0.75 #min rate of doubling
t_range = [12,16] #range of time-steps
sigma = 6 #number of tumor samples
beta = 0.00 #extinction rate

# make samples
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
sample_list = [alphabet[i] for i in range(sigma)]

columner1 = ['mu', 'alpha', 'N', 'n_mets', 'NN_met', 'DynMet']
columner2 = [sample_list[i] for i in range(len(sample_list))]
masterDF = pd.DataFrame(columns=columner1+columner2)

i = 0
for m in tqdm([0.025]):
    for j in tqdm(range(N_clones)):

        pp, sp = compute_priors(number_of_characters, number_of_states, m, mean=1, disp=0.1)

        tree, params = data_gen.generate_simulated_experiment_plasticisity(pp,
            [dropout]*number_of_characters,
            characters=number_of_characters,
            subsample_percentage=0.5,
            dropout=True, 
            sample_list = sample_list, 
            max_mu = max_mu, 
            min_alpha = min_alpha,
            depth = depth, 
            beta = 0,
            )
        
        n_mets, mets, freqs = get_transition_stats(tree.network)
        
        leaves = [n for n in tree.network if tree.network.out_degree(n) == 0]
        meta = pd.DataFrame.from_dict(dict(zip([l.name for l in leaves], [tree.network.nodes[l]['meta'] for l in leaves])), orient='index', columns = ['sample'])
        nn_met = met_utils.compute_NN_metastasis_score(tree.network, meta['sample'], _method = 'allele', verbose=False)
        norm_fitch = met_utils.compute_dynamic_metastasis_score(tree.network, meta['sample'])
        
        masterDF.loc[i, 'mu'] = params['mu']
        masterDF.loc[i, 'alpha'] = params['alpha']
        masterDF.loc[i, 'N'] = params['N']
        masterDF.loc[i, 'n_mets'] = n_mets
        masterDF.loc[i, 'NN_met'] = nn_met
        masterDF.loc[i, 'DynMet'] = norm_fitch
        
        for k in freqs:
            masterDF.loc[i, k] = freqs[k]
        i += 1

masterDF.to_csv("met_sim_df.small.txt", sep='\t')
