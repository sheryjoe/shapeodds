function [fs, D, fdims] = SquareMaps(fs, fdims, N)

% images are not square, make them for isotropic filter design and need
% to report model parameters wrt the original size
fdims_orig = fdims; 
fs_orig    = fs;

is2D = length(fdims)==2;

% for d = 1 : length(fdims_orig)
%     idx{d} = [num2str(1) ':' num2str(fdims_orig(d))] ;
% end

maxDim = max(fdims);
fdims  = maxDim .* ones(size(fdims));
fs     = zeros([fdims N]);

if N == 1
    f_n_orig = fs_orig;
    f_n       = zeros(fdims);
    
    if is2D
        f_n(1:fdims_orig(1), 1:fdims_orig(2)) = f_n_orig(1:fdims_orig(1), 1:fdims_orig(2));
    else
        f_n(1:fdims_orig(1), 1:fdims_orig(2), 1:fdims_orig(3)) = f_n_orig(1:fdims_orig(1), 1:fdims_orig(2), 1:fdims_orig(3));
    end
    
    fs = f_n;
    
else
    for n = 1 : N
        f_n_orig  = GetNthMap(fs_orig, n);
        
        f_n       = zeros(fdims);
        
        if is2D
            f_n(1:fdims_orig(1), 1:fdims_orig(2)) = f_n_orig(1:fdims_orig(1), 1:fdims_orig(2));
        else
            f_n(1:fdims_orig(1), 1:fdims_orig(2), 1:fdims_orig(3)) = f_n_orig(1:fdims_orig(1), 1:fdims_orig(2), 1:fdims_orig(3));
        end
        
        %f_n(idx{:}) = f_n_orig(idx{:});
        fs          = SetNthMap(fs, f_n, n);
    end
end


clear fs_orig

D = prod(fdims);