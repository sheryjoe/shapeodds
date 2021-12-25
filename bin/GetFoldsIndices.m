
function folds_indices = GetFoldsIndices(nSamples, nFolds)

foldSizes = ComputeFoldSizes(nSamples, nFolds);

folds_indices = {};
ind = 0;
for k = 1 : nFolds
    fold_indices = [];
    for ii = 1 : foldSizes(k)
        ind = ind + 1;
        fold_indices(ii) = ind;
    end
    folds_indices{k} = fold_indices;
end