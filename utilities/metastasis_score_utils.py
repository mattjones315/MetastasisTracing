import numpy as np
from . import fitch_parsimony

def compute_tree_complexity(tree):
    
    _leaves = [n for n in tree if tree.out_degree(n) == 0]
    _int_nodes = [n for n in tree if tree.out_degree(n) >= 1]
    
    return len(_leaves) / (len(_int_nodes) + len(_leaves)), len(_leaves), len(tree.nodes)

def scale_counts(T):
    """
    Utility function to scale counts. Takes in a N x 2 contingency table, T, and scales the counts such
    that the column sum of each column is the same.
    """
    
    col_counts = T.sum(axis=0)
    
    scale_fact = col_counts[0] / col_counts[1]
    
    T["LG"] *= scale_fact
    
    return T

def calc_props(T):
    
    col_counts = T.sum(axis=0)
    
    T /= col_counts
    
    return T

def cramers_v(stat, N, k, r): 
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher, 
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    phi2 = stat/N

    phi2corr = max(0, phi2 - ((k-1) * (r-1)) / (N-1))
    rcorr = r - ((r-1)**2)/(N-1)
    kcorr = k - ((k-1)**2)/(N-1)

    v = np.sqrt( phi2corr / min( (kcorr - 1), (rcorr - 1) ))
        
    return v 

def compute_static_metastasis_score(meta_lg, group_var = 'sampleID'):
    """
    Comptues the static metastatic score using the Cramer's V statistic. 

    parameters:
        meta_lg: N x M meta file for a given clonal population of N cells. This meta 
        file can have an arbitrary number of variables.
        group_var: variable by which the static metastatic score will be computed. This 
        must be a column name in the meta_lg object.

    Returns:
        N (the shape of the meta_lg object), Chi-Sq. stat, Log10(Chi-Sq stat), Cramer's V
    """


    query = {}
    for n, g in meta_lg.groupby(group_var):
        query[n] = g.shape[0]


    query = pd.DataFrame.from_dict(query, orient="index")

    table = pd.concat([background, query], axis=1)
    table.fillna(value = 0, inplace=True)
    table.columns = ["Background", "LG"]
    table["Background"] = table["Background"] - table["LG"]

    # scale LG counts
    table = scale_counts(table)

    stat = scs.chi2_contingency(table)[0]
        
    v = cramers_v(stat, np.sum(table.sum()), table.shape[0], table.shape[1])
    
    return meta_lg.shape[0], stat, np.log10(stat), v

def compute_dynamic_metastasis_score(tree, meta):
    """
    Computes the dynamic metastatic score. 

    parameters:
        tree: Networkx object representing the tree. 
        meta: N x 1 Pandas dataframe, mapping each leaf to the meta variable of interest (e.g. tissue ID)

    Returns:
        The dynamic metastatic score -- i.e. the normalized parsimony with respect to the meta variable specified. 
    """

    tree = fitch_parsimony.assign_labels(tree, meta)
    tree = fitch_parsimony.fitch(tree)

    score = fitch_parsimony.score_parsimony(tree)

    return score / len(list(tree.edges()))

    
