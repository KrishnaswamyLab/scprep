%%Aug 7 2017, Jay Stanley 
%%load data
data_dir = '';
sdata_raw = load_10xData(data_dir, 1e15);

%%
sdata = sdata_raw;
%% load barcodes


%% filter out high mtDNA
remove_top_prct = 10;
%mt columns
mt_ind = find(~cellfun('isempty',regexpi(sdata.genes,'mt-')));
M = sdata.data(:,mt_ind);
%normalize by library size
M = bsxfun(@rdivide, M, sdata.library_size);
%sum 
M = mean(M, 2);

pt = prctile(M, 100-remove_top_prct);
cells_keep = M < pt;

sum(cells_keep)
sum(~cells_keep)

%% library size filtering
min_lib = prctile(sdata.library_size, 5);
max_lib = prctile(sdata.library_size, 95);

cells_keep = sdata.library_size >= min_lib ...
    & sdata.library_size <= max_lib ...
    & cells_keep;
%% remove empty genes
genes_keep = sum(sdata.data(cells_keep,:))>10;
sdata.data = sdata.data(:,genes_keep);
filtered_genes = sdata.genes(~genes_keep);

sdata.genes = sdata.genes(genes_keep);
sdata.mpg = sdata.mpg(genes_keep);
sdata.cpg = sdata.cpg(genes_keep);
sdata = sdata.recompute_name_channel_map();

%% downsampling
downsample_size = median(sdata.library_size(cells_keep));
sdata = downsample_molecules(sdata, downsample_size);
sdata = sdata.normalize_data_fix_zero();


