
function neglogPrior = ComputeNegLogPriorForLthVector(wl, sample_dims, priorType, lambda, beta, useSmoothness, useARD, neglogPrior)

if length(sample_dims) == 2
    wl = reshape(wl, sample_dims(1),sample_dims(2));
else
    wl = reshape(wl, sample_dims(1),sample_dims(2),sample_dims(3));
end

if useSmoothness
    switch priorType,
        case  'dirichlet',
            if length(sample_dims) == 2
                [wl_x, wl_y]     = gradient(wl);
                wl_dirchlet      = wl_x.^2 + wl_y.^2;
            else
                [wl_x, wl_y, wl_z]  = gradient(wl);
                wl_dirchlet         = wl_x.^2 + wl_y.^2 + wl_z.^2;
            end
            neglogPrior      = neglogPrior + (lambda/2) * sum(wl_dirchlet(:)); % smoothness
            
        case 'laplacian-square',
            wl_laplace_      = (2*ndims(wl)).*del2(wl);
            
            if length(sample_dims) == 2
                wl_laplace       = zeros(sample_dims(1), sample_dims(2));
                wl_laplace(2:end-1,2:end-1) = wl_laplace_(2:end-1,2:end-1);
            else
                wl_laplace       = zeros(sample_dims(1), sample_dims(2), sample_dims(3));
                wl_laplace(2:end-1,2:end-1,2:end-1) = wl_laplace_(2:end-1,2:end-1,2:end-1);
            end
            neglogPrior      = neglogPrior  + (lambda/2) * sum(wl_laplace(:).^2); % smoothness
    end
end

if useARD
    % add ARD prior part (only for loading vectors)
    neglogPrior      = neglogPrior  + (beta/2) * (wl(:)'*wl(:)); %  ARD
end