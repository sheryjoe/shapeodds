
function [factorsPosterior, gamma, logLikelihood] = Estep(fs, Theta, alpha, options, gamma_init)

% for each label map, optimize for its global variational parameters
% gamma_n = {m_n, V_n}

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
loglikelihood_store    = zeros(1,N);
log_store              = cell(1,N);

computeGradient = 1;
posteriorMean   = zeros(L,N);
posteriorCov    = zeros(L,L,N);
gamma           = zeros(nVariationalParams,N);

% go-over all the training samples to optimize for their latent factors posterior
% approximate distribution
if options.ispar
    
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        myCluster            = parcluster('local');
        myCluster.NumWorkers = options.nworkers;
        parpool('local', options.nworkers);
    end
    
    options.updateFuncParams.L = L;
        
    parfor n = 1 : N        
        % current sample
        f_n  = fs(:,n);
        
        fprintf('\tE-Step (ispar = 1): sample # %d of %d ...\n', n, N);
        
        % initialize global variational parameters of the current sample
        if isempty(gamma_init)
            m_n_init     = Theta.Mu;
            V_n_init     = Theta.Sigma;
            
            % pack-in the initial parameters vector
            gamma_n_init = PackVariationalParams(m_n_init, V_n_init);
        else
            gamma_n_init = gamma_init(:,n);
        end
        
        % optimize global variational parameters by maximizing the sample's
        % log-likelihood lower bound given the current model parameters (Theta) and the
        % local variational bound parameters (alpha)
        tic
        [gamma_n, neglogLik_n, ~, output] = minFunc2(@SampleNegLogLikelihoodBound, gamma_n_init, options, Theta, f_n, alpha, computeGradient);
        elapsedTime = toc;
        
        fprintf('Converged in %d iterations and took %f seconds ...\n',output.iterations, elapsedTime);
        
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
        
        fprintf('\tE-Step (ispar = 0): sample # %d of %d ...\n', n, N);
        
        % initialize global variational parameters of the current sample
        if isempty(gamma_init)
            m_n_init     = Theta.Mu;
            V_n_init     = Theta.Sigma;
            
            % pack-in the initial parameters vector
            gamma_n_init = PackVariationalParams(m_n_init, V_n_init);
        else
            gamma_n_init = gamma_init(:,n);
        end
        
        % optimize global variational parameters by maximizing the sample's
        % log-likelihood lower bound given the current model parameters (Theta) and the
        % local variational bound parameters (alpha)
        tic
        options.updateFuncParams.L = L;
        [gamma_n, neglogLik_n, ~, output] = minFunc2(@SampleNegLogLikelihoodBound, gamma_n_init, options, Theta, f_n, alpha, computeGradient);
        elapsedTime = toc;
        
        fprintf('Converged in %d iterations and took %f seconds ...\n',output.iterations, elapsedTime);
        
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

