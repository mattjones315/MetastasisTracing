import numpy as np
import pandas as pd
from tqdm import tqdm
import sys


def iterate_doublings(cells, alpha):
    new_cells = []
    for cell in cells:
        new_cells.append(cell)
        if np.random.random() < alpha: # cell divides!
            new_cells.append(cell)
    return new_cells

def iterate_extinction(cells, beta):
    new_cells = []
    for cell in cells:
        if (np.random.random() > beta) or (len(cells)<=10): # cell survives!
            new_cells.append(cell)
    return new_cells

def iterate_metastasis(cells, mu, n_mets, sample_list):
    new_cells = []
    for cell in cells:
        if np.random.random() >= mu: 
            new_cells.append(cell)
        else:
            temp_sample_list = sample_list.copy()
            temp_sample_list.remove(cell)
            new_sample = np.random.choice(temp_sample_list) # cell metastasizes!
            new_cells.append(new_sample)
            n_mets+=1
    return new_cells, n_mets

def simulate_counts(sample_list, N_clones, max_mu, min_alpha, t_range, beta):

    columner1 = ['mu', 'alpha', 't', 'N', 'n_mets']
    columner2 = [sample_list[i] for i in range(len(sample_list))]
    masterDF = pd.DataFrame(columns=columner1+columner2)

    for iteration in tqdm(range(0,N_clones), desc='iterating over clones'):
        
        cells_list = ['A']
        
        # choose parameters randomly:
        mu = max_mu*np.random.random() # probability of metastasis per time-step (between 0 and 0.3)
        alpha = min_alpha+((1-min_alpha)*np.random.random()) # probability that cell will double per time-step (between 0.75 and 1.0)
        t = np.random.randint(t_range[0],high=t_range[1]) # number of time-steps in simulation
        
        # run iterations
        n_mets = 0
        for timestep in range(0,t):
            cells_list = iterate_doublings(cells_list, alpha)
            if beta > 0: cells_list = iterate_extinction(cells_list, beta)
            if mu > 0: cells_list, n_mets = iterate_metastasis(cells_list, mu, n_mets, sample_list)
        
        # count cells per sample
        counts = []
        for sample in sample_list:
            counts.append(cells_list.count(sample))
        
        # update master
        col_i1 = [mu, alpha, t, len(cells_list), n_mets]
        col_i2 = [counts[i] for i in range(len(counts))]
        masterDF.loc[len(masterDF), :] = col_i1+col_i2
            
    # calculate the normalized met count
    masterDF['Norm_mets'] = masterDF['n_mets'].div(masterDF['N'])
    masterDF = masterDF.sort_values(['Norm_mets'], ascending=False).reset_index(drop=True)

    return masterDF