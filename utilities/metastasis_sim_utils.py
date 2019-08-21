import numpy as np


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