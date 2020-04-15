
using MatrixMarket;
using GLM;
using DataFrames;
using CSV;
using Statistics;
using Distributions;
using MixedModels;
using Tables;


### Many functions below taken from https://github.com/ThomsonMatt/CompBioClass#loading-mtx-files-in-julia
function read_csc(pathM::String)
    X = MatrixMarket.mmread(string(path10x, "matrix.mtx"));
    Float64.(X)
end

function read_barcodes(tsvPath::String)
    f=open(tsvPath)
    lines=readlines(f)
    a=String[]
    for l in lines
        push!(a,l)
    end
    close(f)
    return a
end

function read_genes(tsvPath::String)
    f=open(tsvPath)
    lines=readlines(f)
    a=String[] #Array{}
    for l in lines
        push!(a,uppercase(split(l,"\t")[2]))
    end
    close(f)
    return a
end

function sort_array_by_list(arr, _list)

    order = [];
    for bc in _list
        i = findall(x->x == bc, arr)[1]
        push!(order, i)
    end
    order
end

function calc_log2fc(up, dwn)

    fc = (0.01 + mean(up)) / (0.01 + mean(dwn));
    log2(fc)

end

count_nnz(x) = count(i -> (i>0), x)

println("reading in data")

path10x = "/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/ALL_Samples/GRCh38/";
meta_fp = "/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/LG_meta_fixed.txt"
# meta_fp = "/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/LG_meta.shuffled.txt"; # let's look at shuffled controls 
expr = Array(read_csc(string(path10x, "matrix.mtx")));
barcodes = read_barcodes(string(path10x, "barcodes.tsv"));
genes = read_genes(string(path10x, "genes.tsv"));
println(string("read in a gene expression matrix with ", length(barcodes), " cells and ", length(genes), " genes"))

println("filtering out apoptotic cells")
mito_genes = [name for name in genes if startswith(name, "MT-")];
mito_inds = findall(i -> i in mito_genes, genes);
mito_prop = mapslices(i -> sum(i[mito_inds,:]) / sum(i), expr, dims=1);
keep_cells = [i <= 0.2 for i in mito_prop][1,:];
println(string("filtering out ", size(expr)[2] - sum(keep_cells), " cells"))
expr = expr[:, keep_cells];
barcodes = barcodes[keep_cells];

println("reading in and filtering by meta data entries")
meta_data = CSV.read(meta_fp, delim='\t');

# drop out rows that have missing values
meta_data = meta_data[completecases(meta_data),:];

# only predict on top and bot percentiles
# top_perc = 0.21;
# bot_perc = 0.004; 
# meta_data = filter(row->((row.scTreeMetRate >= top_perc) | (row.scTreeMetRate <= bot_perc)), meta_data);

keep_cells = intersect(meta_data.cellBC, barcodes);
meta_filtered = filter(row->row.cellBC in keep_cells, meta_data);

order = sort_array_by_list(barcodes, meta_filtered.cellBC);
expr = expr[:, order];

# calculate size factors
size_factors = [i for i in mapslices(i -> sum([x > 0 for x in i]) / length(genes), expr, dims=1)[1, :]];

println("normalizing library counts")
norm_factor = median(mapslices(i -> sum(i), expr, dims=1));
expr_norm = mapslices(i -> (i * norm_factor) / sum(i), expr, dims=1);


println("filtering out lowly expressed genes")
nnz = mapslices(count_nnz, expr, dims=2);
threshold = 0.1*size(expr)[2]
keep_ii = map(i->i[1], findall(x -> x>threshold, nnz));

genes = genes[keep_ii];
expr_filt = expr[keep_ii,:];
expr_norm_filt = expr_norm[keep_ii,:];

# center dynamic met DynamicMetScore
mu = mean(meta_filtered.scTreeMetRate);
meta_filtered.scTreeMetRate = [(i - mu) for i in meta_filtered.scTreeMetRate];

refactor_met_score(x) = ifelse(x >= 0, 1, 0);
y = [refactor_met_score(i) for i in meta_filtered.scTreeMetRate];
lineage_group = [string(i) for i in meta_filtered.LineageGroup];

cell_counts = [i for i in size_factors];
X = transpose(expr_filt);
Xn = transpose(expr_norm_filt);

up_cells = Xn[y .== 1,:];
dwn_cells = Xn[y .== 0,:];

betas = [];
pvalues = [];
genes_tested = [];
test_type = [];
fold_changes = [];

println(string("continuing with model of ", size(X)[1], " cells and ", size(X)[2], " genes."))

for i in 1:size(X)[2]

    df = DataFrame(x = (X[:, i] .+ 1), y = string.(y), lg = lineage_group, sz = cell_counts);

    try

        # poisson regression
        m0 = glm(@formula(x ~ sz), df, Poisson(), LogLink());
        m1 = glm(@formula(x ~ sz + y), df, Poisson(), LogLink());
        push!(test_type, "Poisson");

        l0 = loglikelihood(m0);
        l1 = loglikelihood(m1);
        lr = -2 * (l0 - l1);
        pval = ccdf(Chisq(1), lr);

        fc = calc_log2fc(up_cells[:,i], dwn_cells[:,i]);

        push!(betas, coef(m1)[3]);
        push!(pvalues, pval);
        push!(genes_tested, genes[i]);
        push!(fold_changes, fc)

    catch e
      
        println(string("encountered error at gene ", genes[i]))
        
    end

    

end 

res_df = DataFrame(genes = genes_tested, betas = betas, pvalues = pvalues, test=test_type, log2fc = fold_changes);
CSV.write(open("/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/ALL_Samples/poisson_reg_de.ALL_fixed.txt", "w"), res_df, delim = "\t");
