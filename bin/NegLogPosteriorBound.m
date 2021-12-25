
function [neglogPosterior, neglogPosteriorGradient] = NegLogPosteriorBound(Wvec, Theta, fs, gamma, alpha, priorType, Epsi, ...
    sample_dims, useARD, useSmoothness, usePriorForObjective, PriorFactor, computeGradient, ispar, nworkers)

%--------------------------

% Theta = {Mu, Sigma/SigmaInv, W, w0}: model parameters
% f_n: the nth label map
% gamma_n = {m_n, V_n}: global variational parameters for the nth sample
% alpha : local variational bound parameters (piecewise quadratic bound)

% number of samples N and observed space dimensionality D
[D,N] = size(fs);

% latent space dimensionality
L     = size(Theta.Mu,1);

% the current bias and loading matrix
Wtilde = reshape(Wvec, [D,1 + L]); % the first column is the offset vector

% model prior hyperparameters
lambdas  = Epsi.lambdas; % 1 x 1 + L
betas    = [0 Epsi.betas];   % 1 x L

% update model parameters with the current loading matrix and offset vector
w0 = Wtilde(:,1);
W  = Wtilde(:,2:end);

% evaluate the likelhood of the model and the gradient of the objective function
% given the current model parameters and the posterior distribution parameters
logLikelihood  = zeros(1,N);

% gradient wrt w0 (offset vector)
gw0 = zeros(D,N);

% gradient wrt loading matrix W
gW  = zeros(D,L,N);

% compute loglikelihood bound and its gradient wrt model parameters
if ispar
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        myCluster            = parcluster('local');
        myCluster.NumWorkers = nworkers;
        parpool('local', nworkers);
    end
    
    parfor n = 1 : N
        [logLikelihood(n), gw0(:,n), gW(:,:,n)] = ComputeSampleLogLikelihoodBoundAndGradient(fs(:,n), gamma(:,n), alpha, W, w0, computeGradient,...
            logLikelihood(n), gw0(:,n), gW(:,:,n));
    end
        
else
    for n = 1 : N
        [logLikelihood(n), gw0(:,n), gW(:,:,n)] = ComputeSampleLogLikelihoodBoundAndGradient(fs(:,n), gamma(:,n), alpha, W, w0, computeGradient, ...
            logLikelihood(n), gw0(:,n), gW(:,:,n));
    end
    
end
neglogLikelihood = -sum(logLikelihood);

% evaluate the prior contribution given the current model paramters
neglogPrior = 0;

if usePriorForObjective
    neglogPrior_l = zeros(1,1 + L);
    
    if ispar
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            myCluster            = parcluster('local');
            myCluster.NumWorkers = nworkers;
            parpool('local', nworkers);
        end
        
        parfor l = 1 : 1 + L
            wl = Wtilde(:,l);
            
            if l == 1
                neglogPrior_l(l) = ComputeNegLogPriorForLthVector(wl, sample_dims, priorType, lambdas(l), betas(l), useSmoothness, false, neglogPrior_l(l));
            else
                neglogPrior_l(l) = ComputeNegLogPriorForLthVector(wl, sample_dims, priorType, lambdas(l), betas(l), useSmoothness, useARD, neglogPrior_l(l));
            end
        end
        
    else
        for l = 1 : 1 + L
            wl = Wtilde(:,l);
            
            if l == 1
                neglogPrior_l(l) = ComputeNegLogPriorForLthVector(wl, sample_dims, priorType, lambdas(l), betas(l), useSmoothness, false, neglogPrior_l(l));
            else
                neglogPrior_l(l) = ComputeNegLogPriorForLthVector(wl, sample_dims, priorType, lambdas(l), betas(l), useSmoothness, useARD, neglogPrior_l(l));
            end
        end
        
    end
    neglogPrior = sum(neglogPrior_l); 
end

% toc
% tic
% the final value for the negative log posterior
% NOTE: this value only affects time step computation so no need to be
% driven by the prior, especially in case of very huge lambda values, that
% is if the optimization at some point leads to large weight for lambda,
% allowing the time step to be driven by the data term will pull the
% estimate to comply with the data-term while still applying the gradient
% based on the data and prior terms
if usePriorForObjective
    % note: this factor should be constant for the whole optimization
    % process to guarantee monotonic decrease of the objective function
    neglogPosterior = neglogLikelihood + PriorFactor * neglogPrior;
else
    neglogPosterior = neglogLikelihood;
end

if computeGradient
    
    % compute the gradients
    neglogPosteriorGradient = zeros(D, 1 + L); % the first column is the offset vector
    
    neglogPosteriorGradient(:,1)     = -sum(gw0,2);
    neglogPosteriorGradient(:,2:end) = -sum(gW,3);
    
    if useARD
        % add ARD prior part (only for loading vectors)
        for l = 2 : 1 + L
            wl                           = Wtilde(:,l);
            neglogPosteriorGradient(:,l) = neglogPosteriorGradient(:,l) + betas(l) .* wl;
        end
    end
    
    % reshape to be 1d vector
    neglogPosteriorGradient = neglogPosteriorGradient(:);
    
else
    neglogPosteriorGradient = 0;
end