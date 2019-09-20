MetastasisTracing
====================
Scripts & Notebooks for the analysis of dynamic lineage tracing of lung cancer metastasis.

This repository serves as a common base for all analysis notebooks (stored in the main directory) and the utility scripts (stored in `utilities`).

Perhaps most useful in the `utilities` directory are:
    
- `metastasis_score_utils`: functions for scoring metastatic ability. Pertinent functions include *compute_static_metastasis_score* and *compute_dynamic_metastasis_score*.
- `metastasis_sim_utils`: functions for simulating metastasis
- `validate_trees`: procedures for looking at QC metrics of trees (e.g. the relationship between indel edit distance & phylogenetic distance of all cells)

Each notebook corresponds to a module in our analysis:
	
* `Metastasis_Simulator.ipynb`: simulate metastasis over various tissues and extract out tissue distributions and observed metastatic events.
* `Score_Metastasis.ipynb`: filter out clones & score metastatic ability.
