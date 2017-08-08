%%Aug 7 2017, Jay Stanley 
%%load data
data_dir = '/data/lab/DataSets/ParkLab/corticalorganoids';
sdata_raw = load_10xData(data_dir, 1e15);

%%
sdata = sdata_raw;
%% load barcodes

barcode_file = data_dir+"/barcodes.tsv";
barcodes = textscan(fopen(barcode_file),'%s');
barcodes = barcodes{1};
experiment_idxs = ones(size(barcodes,1),1);
%split barcodes into cell types
for i = 1:size(barcodes, 1)
    cur_code = strsplit(barcodes{i},'-');
    experiment_idxs(i) = str2double(cur_code(2));
end

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

%% struct of experiments w/ filtered cells
experiment_idxs = experiment_idxs(cells_keep);
experiment_field = 'name';
%order is key here
experiment_labels = {'ehCO_1'; 'ehMGEO_1'; 'ehMGEO_2'; 'ehCO_2'; 'lhMGEO_1'; 'lhCO_1'; 'lhMGEO_2'; 'lhCO_2'};
experiments = struct(experiment_field, experiment_labels);
for i = 1:8
    cur = experiment_idxs==i;
    experiments(i).num = nnz(cur);
    experiments(i).indices = find(cur);
    experiments(i).data = sdata.data(cur,:);
    experiments(i).library_size = sdata.library_size(cur);
    experiments(i).genes = sdata.genes;
    experiments(i).cells = experiments(i).data(cur);
end

