function [sdata, geneID_map] = load_10xData(data_dir,n_cells)
% This function will load scRNA-seq data output from the 10x Genomics
% Cell Ranger pipeline. data_dir should be the path to the directory 
% containing three files: barcodes.tsv, genes.tsv, and matrix.mtx.
% n_cells (optional) is the number of cells to randomly subsample 
% from the matrix. Subsampling reduces runtime and computational 
% load, with a default maximum of 40,000 cells. 

tic

if isempty(data_dir)
    data_dir = './';
elseif data_dir(end) ~= '/'
    data_dir = [data_dir '/']; 
end

filename_dataMatrix = [data_dir 'matrix.mtx'];
filename_genes = [data_dir 'genes.tsv'];

% Read in gene expression matrix (sparse matrix)
% Rows = genes, columns = cells
fprintf('LOADING\n')
dataMatrix = mmread(filename_dataMatrix);
fprintf('  Data matrix (%i cells x %i genes): %s\n', ...
        size(dataMatrix'), ['''' filename_dataMatrix '''' ])

% Read in row names (gene names / IDs)
dataMatrix_genes = table2cell( ...
                   readtable(filename_genes, ...
                             'FileType','text','ReadVariableNames',0));
                                    
% Remove empty cells
col_keep = any(dataMatrix,1);
dataMatrix = dataMatrix(:,col_keep);
fprintf('  Removed %i empty cells\n', full(sum(~col_keep)))

% Remove empty genes
genes_keep = any(dataMatrix,2);
dataMatrix = dataMatrix(genes_keep,:);
dataMatrix_genes = dataMatrix_genes(genes_keep,:);
fprintf('  Removed %i empty genes\n', full(sum(~genes_keep)))

% Store gene name/ID map
geneID_map = containers.Map(dataMatrix_genes(:,1), dataMatrix_genes(:,2));

% Subsample cells, max 40000 cells
if ~exist('n_cells','var') || isempty(n_cells)
    n_cells = 40000;
end

if n_cells < size(dataMatrix,2)
    fprintf('  Subsample cells, N = %i\n',n_cells)
    col_keep = randsample(size(dataMatrix,2), n_cells);
    dataMatrix = dataMatrix(:,col_keep);
end

% Convert to sdata object
% Rows = cells, columns = genes
sdata = scRNA_data('data_matrix', full(dataMatrix'), ... 
                   'gene_names', dataMatrix_genes(:,2)); % Use gene IDs (column 1)

toc
fprintf('\n%i x %i (cells x genes) MATRIX\n', size(sdata.data))
end