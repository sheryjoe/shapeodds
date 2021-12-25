
function [factorsPosterior, gamma, logLikelihood, outlier_masks, taus] = RobustEstep(fs, Theta, alpha, options, gamma_init)

% for each label map, optimize for its global variational parameters
% gamma_n = {m_n, V_n}

MaxRobustIter        = options.MaxRobustIter;
TauContractionFactor = options.TauContractionFactor ;
RhoFuncType          = options.RhoFuncType;

% observed space dimensionality and  number of samples
% note that label maps were already vectorized
[D, N] = size(fs);

% latent space dimensionality
L = size(Theta.Mu,1);

% placeholders and initializations
% number of actual degrees of freedom = L (for mean) plus the number of
% elements in the upper triangular matrix of the covariance matrix
nVariationalParams = length(find(triu(ones(L)))) + L;

% initializations
loglikelihood_store    = zeros(N);
log_store              = cell(N);

computeGradient = 1;
posteriorMean       = zeros(L,N);
posteriorCov        = zeros(L,L,N);
gamma               = zeros(nVariationalParams,N);

outlier_masks      = [];
taus               = [];

% go-over all given samples to optimize for their latent factors posterior
% approximate distribution
if options.ispar
    
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        myCluster            = parcluster('local');
        myCluster.NumWorkers = options.nworkers;
        parpool('local', options.nworkers);
    end
    
    options.updateFuncParams.L = L;
    
    outlier_masks = [];
    taus          = [];
    ispar         = options.ispar;
    Mu            = Theta.Mu;
    Sigma         = Theta.Sigma;
    
    parfor n = 1 : N
        
        % current sample
        f_n  = fs(:,n);
        
        fprintf('Robust E-Step (ispar = %d): sample # %d of %d ...\n', ispar, n, N);
        
        % initialize global variational parameters of the current sample
        if isempty(gamma_init)
            m_n_init     = Mu;
            V_n_init     = Sigma;
            
            % pack-in the initial parameters vector
            gamma_n_init = PackVariationalParams(m_n_init, V_n_init);
        else
            gamma_n_init = gamma_init(:,n);
        end
        
        % assume no outliers, optimize for gamma using all pixels of
        % the current sample
        % optimize global variational parameters by maximizing the sample's
        % log-likelihood lower bound given the current model parameters (Theta) and the
        % local variational bound parameters (alpha)
        
        gamma_n     = gamma_n_init;
        neglogLik_n = 1e+10;
        [~,~,~, E_gamma_x] = SampleRobustNegLogLikelihoodBound(gamma_n, Theta, f_n, alpha, 0, RhoFuncType, computeGradient);
        tau                = max(E_gamma_x); % starting with all inliers
        
        for iter = 1 : MaxRobustIter
            gamma_n_old     = gamma_n;
            neglogLik_n_old = neglogLik_n;
            
            fprintf('\tRobust E-Step (ispar = %d): sample # %d of %d, robust iter = %d, tau = %2.3f ... \n', ispar, n, N, iter, tau);
            
            tic
            [gamma_n, neglogLik_n, ~, output] = minFunc2(@SampleRobustNegLogLikelihoodBound, gamma_n, options, Theta, f_n, alpha, tau, RhoFuncType, computeGradient);
            elapsedTime = toc;
            fprintf('converged in %d iterations and took %f seconds ...\n',output.iterations, elapsedTime);
            
            [E_R, gradE_R, outlier_mask, E_gamma_x] = SampleRobustNegLogLikelihoodBound(gamma_n, Theta, f_n, alpha, tau, RhoFuncType, computeGradient);
            
            % check for convergence
            delta = max(abs(gamma_n_old  - gamma_n)  / (sqrt(eps)+max(max(abs(gamma_n)))));
            
            if delta < options.TolX
                break;
            elseif abs(neglogLik_n - neglogLik_n_old)  < options.TolFun
                break;
            end
            
            if tau < 1e-5
                break;
            end
            
            tau = tau * TauContractionFactor;
        end
        
        outlier_masks(:,n) = outlier_mask;
        taus(n)            = tau;
        
        loglikelihood_store(n) = -neglogLik_n;
        log_store{n}           = output;
        
        % unpack the parameters
        [m_n, V_n] = UnpackVariationalParams(gamma_n, L);
        
        posteriorMean(:,n)   = m_n;
        posteriorCov(:,:,n)  = V_n;
        
        % store values to warmstart next inference (e-step) run
        gamma(:,n)           = gamma_n;
    end
    
else
    for n = 1 : N
        % current sample
        f_n  = fs(:,n);
        
        fprintf('Robust E-Step (ispar = %d): sample # %d of %d ...\n', options.ispar, n, N);
        
        % initialize global variational parameters of the current sample
        if isempty(gamma_init)
            m_n_init     = Theta.Mu;
            V_n_init     = Theta.Sigma;
            
            % pack-in the initial parameters vector
            gamma_n_init = PackVariationalParams(m_n_init, V_n_init);
        else
            gamma_n_init = gamma_init(:,n);
        end
        
        % assume no outliers, optimize for gamma using all pixels of
        % the current sample
        % optimize global variational parameters by maximizing the sample's
        % log-likelihood lower bound given the current model parameters (Theta) and the
        % local variational bound parameters (alpha)
        
        options.updateFuncParams.L = L;
        gamma_n     = gamma_n_init;
        neglogLik_n = 1e+10;
        [~,~,~, E_gamma_x] = SampleRobustNegLogLikelihoodBound(gamma_n, Theta, f_n, alpha, 0, RhoFuncType, computeGradient);
        tau                = max(E_gamma_x); % starting with all inliers
        
        for iter = 1 : MaxRobustIter
            gamma_n_old     = gamma_n;
            neglogLik_n_old = neglogLik_n;
            
            fprintf('\tRobust E-Step (ispar = %d): sample # %d of %d, robust iter = %d, tau = %2.3f ... \n', options.ispar, n, N, iter, tau);
            
            tic
            [gamma_n, neglogLik_n, ~, output] = minFunc2(@SampleRobustNegLogLikelihoodBound, gamma_n, options, Theta, f_n, alpha, tau, RhoFuncType, computeGradient);
            elapsedTime = toc;
            fprintf('converged in %d iterations and took %f seconds ...\n',output.iterations, elapsedTime);
            
            [E_R, gradE_R, outlier_mask, E_gamma_x] = SampleRobustNegLogLikelihoodBound(gamma_n, Theta, f_n, alpha, tau, RhoFuncType, computeGradient);
            
            % check for convergence
            delta = max(abs(gamma_n_old  - gamma_n)  / (sqrt(eps)+max(max(abs(gamma_n)))));
            
            if delta < options.TolX
                break;
            elseif abs(neglogLik_n - neglogLik_n_old)  < options.TolFun
                break;
            end
            
            if tau < 1e-5
                break;
            end
            
            tau = tau * TauContractionFactor;
        end
        
        outlier_masks(:,n) = outlier_mask;
        taus(n)             = tau;
        
        loglikelihood_store(n) = -neglogLik_n;
        log_store{n}           = output;
        
        % unpack the parameters
        [m_n, V_n] = UnpackVariationalParams(gamma_n, L);
        
        posteriorMean(:,n)   = m_n;
        posteriorCov(:,:,n)  = V_n;
        
        % store values to warmstart next inference (e-step) run
        gamma(:,n)           = gamma_n;
    end
   
end


% report output
logLikelihood                        = sum(loglikelihood_store(:));
factorsPosterior.posteriorMean       = posteriorMean;
factorsPosterior.posteriorCov        = posteriorCov;
factorsPosterior.loglikelihood_store = loglikelihood_store;
factorsPosterior.optlog_store        = log_store;

