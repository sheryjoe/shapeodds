
function [Theta, Epsi, gamma, L, loadingIdx] = PruneOutIrrelevantLatentDimensions(Theta, Epsi, gamma, L, varianceThreshold)

if L == 0 % only estimating bias
    loadingIdx = 0;
    return;
end

if ~isempty(gamma)
    N = size(gamma,2);
else
    N = 0;
end

L_orig     = L;     L     = [];
gamma_orig = gamma; gamma = [];

% prune the irrelevant dimensions in the current model paramters
% prune out non-informative factor loadings

% to avoid over-prunning, the minimum latent dimension is related to portion of data that the
% current component is responsible for ...
%minLatentDim = ceil(Theta.pi .* N) - 1;
%maxLatentDim = N-1;
%minLatentDim = max(1, ceil(maxLatentDim * 0.1)) ; % 10% of the max allowed latent dimensionality with min of one loading vector (till we allow death moves)
minLatentDim = 1;

% current component variances
variances = 1./(Epsi.betas + eps);

% sort them according to the variance to make sure that we are
% retaining the most relevant ones
idx       = 1 : length(variances);
variances_idx = [variances(:) idx(:)];
variances_idx = sortrows(variances_idx, -1);
idxSorted   = variances_idx(:,2);
variancesSorted = variances_idx(:,1);

% loadings to be retrains that those whose variance exceeds the given
% threshold
nFactors   = sum(variancesSorted > varianceThreshold);

% make sure that we are not over-pruning
nFactors   = max(minLatentDim, nFactors);

% loading indices
loadingIdx = idxSorted(1:nFactors);

% new dimensionality
L = nFactors;

% prune out model parameters
Theta.Mu       = Theta.Mu(loadingIdx);
Theta.Sigma    = Theta.Sigma(loadingIdx,loadingIdx);
Theta.invSigma = inv(Theta.Sigma);
Theta.W        = Theta.W(:, loadingIdx);

% prune out hyperparameters
Epsi.betas     = Epsi.betas(loadingIdx);
Epsi.lambdas   = Epsi.lambdas([1 loadingIdx'+1]); % the first one related to w0

% prune out latent factor variational parameters
if ~isempty(gamma_orig)
    for n = 1 : N
        gamma_n = gamma_orig(:,n);
        
        % unpack the parameters
        [m_n, V_n] = UnpackVariationalParams(gamma_n, L_orig);
        
        m_n = m_n(loadingIdx);
        V_n = V_n(loadingIdx,loadingIdx);
        
        % pack-in the pruned parameters vector
        gamma_n = PackVariationalParams(m_n, V_n);
        
        gamma(:,n) = gamma_n;
    end
end