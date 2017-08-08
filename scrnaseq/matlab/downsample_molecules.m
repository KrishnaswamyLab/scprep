function obj = downsample_molecules(obj, new_lib_size)

[num_cells, num_genes] = size(obj.data);
data_new = zeros(num_cells, num_genes);

for I = 1:num_cells
    I
    cell_vector = obj.data(I,:);
    curr_lib_size = sum(cell_vector);
    if curr_lib_size > new_lib_size
        mol_ind = randperm(curr_lib_size,new_lib_size);
        bins = [0 cumsum(cell_vector)+1];
        binCounts = histc(mol_ind,bins); binCounts(end) = [];
        data_new(I,:) = binCounts;
    else
        warning('cell has fewer than given lib. size molecules')
        data_new(I,:) = cell_vector;
    end
end

obj.data = data_new;