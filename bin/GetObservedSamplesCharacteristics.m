function [D, N, fdim] = GetObservedSamplesCharacteristics(fs)

% observed space dimensionality
fs_size = size(fs);
D       = prod(fs_size(1:end-1)); 

% the last dimension is the number of samples
N       = fs_size(end);

% sample dimension
fdim    = fs_size(1:end-1);