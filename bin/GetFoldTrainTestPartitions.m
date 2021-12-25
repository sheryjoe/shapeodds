
function [Ftrain, Ftest] = GetFoldTrainTestPartitions (fs, k, folds_indices)

nFolds = length(folds_indices);

% current testing set is all label maps in the kth (current) fold
Ftest = fs(:, folds_indices{k});

% current training set is the remaining label maps
Ftrain = [];
for notk = 1 : nFolds
    if notk == k % don't include the current fold in the training set
        continue
    end
    
    Ftrain = [Ftrain fs(:, folds_indices{notk})];
end