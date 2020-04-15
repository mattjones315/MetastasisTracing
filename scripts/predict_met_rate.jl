using MatrixMarket;
using GLMNet;
using DataFrames;
using CSV;
using Statistics;


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
        push!(a,uppercase(l))
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

count_nnz(x) = count(i -> (i>0), x)

println("reading in data")

path10x = "/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/ALL_Samples/GRCh38/";
meta_fp = "/data/yosef2/users/mattjones/projects/metastasis/JQ19/5k/RNA/LG_meta.txt";
expr = Array(read_csc(string(path10x, "matrix.mtx")));
barcodes = read_barcodes(string(path10x, "barcodes.tsv"));
genes = read_genes(string(path10x, "genes.tsv"));

println("filtering out lowly expressed genes")

# filter out lowly expressed genes
nnz = mapslices(count_nnz, expr, dims=2);
threshold = 0.01*length(barcodes);
keep_ii = map(i->i[1], findall(x -> x>threshold, nnz));

genes = genes[keep_ii];
expr_filt = expr[keep_ii,:];

# writing out filtered gene list 
file = open("gene_list_threshfilt.5k.txt", "w")
for i in genes
    println(file, i)
end
close(file)

meta_data = CSV.read(meta_fp, delim='\t');

keep_cells = intersect(meta_data.cellBC, barcodes);
meta_filtered = filter(row->row.cellBC in keep_cells, meta_data);

# drop out rows that have missing values
meta_filtered = meta_filtered[completecases(meta_filtered),:];

order = sort_array_by_list(barcodes, meta_filtered.cellBC);
expr_ordered = expr_filt[:, order];

# normalize counts of library
norm_factor = median(mapslices(i -> sum(i), expr_ordered, dims=1));
expr_norm = mapslices(i -> (i * sum(i)) / norm_factor, expr_ordered, dims=1);


X = log.(transpose(expr_ordered) .+ 1.);
y = log.(0.0001 .+ meta_filtered.DynamicMetScore);

cv = glmnetcv(X, y);
best_fit = argmin(cv.meanloss)

betas = cv.path.betas[:, best_fit];

file = open("lasso_reg_betas.ALL.txt", "w");
for b in betas
    println(file, b);
end
close(file)

# println("Running lasso for each lineage group")
# for i in 1:100
#     if i in meta_filtered.LineageGroup
#         try
#             println(i)
#             sub_meta = filter(row->row.LineageGroup == i, meta_filtered);

#             # reorder expression data frame to line up observations
#             order = sort_array_by_list(barcodes, sub_meta.cellBC);
#             expr_ordered = expr_filt[:, order];

#             # specify X and y 
#             X = log.(transpose(expr_ordered) .+ 1.);
#             y = log.(0.0001 .+ sub_meta.DynamicMetScore);

#             # run glm predicting metastatic rate 
#             cv = glmnetcv(X, y)

#             best_fit = argmin(cv.meanloss)

#             # get best fit's betas
#             betas = cv.path.betas[:, best_fit]
            
#             file = open(string("lasso_reg_betas.", i, ".txt"), "w")
#             for b in betas
#                 println(file, b)
#             end
#             close(file)
#         catch e
#             println("glmnet encountered an error")
#         end
#     end
# end



