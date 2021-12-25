
function [Z, Theta, Epsi, resStruct] = EstimateShapeOdds(fs, shapeoddsOptions)

% Inputs:
% fs : a set of label maps [fdims N]
% shapeoddsOptions: optimizatin options structure (use FillInShapeOddsDefaultOptions
%              for default values

% Outputs:
% Z: latent variables (posterior mean of the given samples)
% Theta: model parameters (offset vector w0 and loading vectors {w_l})
% Epsi: hyperparamters (smoothness parameters lambda_l (for l = 0 : L) and
%       ard parameters beta_l (for l = 1 : L)
% resStruct: contains all detailed output including options for E and M
% steps and their logs

%% make sure that we have all mandatory options for sofa model training
[shapeoddsOptions, halt, halt_msg] = ParseAndCheckShapeOddsOptions(shapeoddsOptions);

if halt
    fprintf('%s ... \n', halt_msg);
    return
end

shapeoddsOptions

%% get the settings we need from the options structure
L              = shapeoddsOptions.L;
maxIter        = shapeoddsOptions.maxIter;
priorType      = shapeoddsOptions.priorType;
isFA           = shapeoddsOptions.isFA;
TolFun         = shapeoddsOptions.TolFun;
TolX           = shapeoddsOptions.TolX;
epsilon        = shapeoddsOptions.epsilon;

saveEMIters    = shapeoddsOptions.saveEMIters;
saveEMFinal    = shapeoddsOptions.saveEMFinal;
out_prefix     = shapeoddsOptions.out_prefix;

mstep_display  = shapeoddsOptions.mstep_display;

EOptimizationMethod = shapeoddsOptions.EOptimizationMethod;
MOptimizationMethod = shapeoddsOptions.MOptimizationMethod;
fsTest              = shapeoddsOptions.fsTest;

useARD              = shapeoddsOptions.useARD;
useSmoothness       = shapeoddsOptions.useSmoothness;
NumberOfInits       = shapeoddsOptions.NumberOfInits;

ispar               = shapeoddsOptions.ispar;
nworkers            = shapeoddsOptions.nworkers;

% initial smoothness and ARD priors
% ARD: if initialized with small values to imply that initially all loadings are relevant
InitBeta   = shapeoddsOptions.InitBeta ;
InitLambda = shapeoddsOptions.InitLambda;

% enable prior to affect time step computation
usePriorForObjectiveInInit         = shapeoddsOptions.usePriorForObjectiveInInit;
usePriorForObjectiveInOptimization = shapeoddsOptions.usePriorForObjectiveInOptimization;

if usePriorForObjectiveInInit
    InitPriorFactor                    = shapeoddsOptions.InitPriorFactor; % constant prior factor for init
else
    InitPriorFactor                    = 0;
end

if usePriorForObjectiveInOptimization
    % decaying prior factor for optimization
    RegularizationContractiveFactor = shapeoddsOptions.RegularizationContractiveFactor;
    StartingRegularization          = shapeoddsOptions.StartingRegularization;
    EndingRegularization            = shapeoddsOptions.EndingRegularization;
end

% for prunning irrelevant factors
PruneIrrelevantLatentDimensions = shapeoddsOptions.PruneIrrelevantLatentDimensions;
LoadingVarianceThreshold        = shapeoddsOptions.LoadingVarianceThreshold;

RandnFactor              = shapeoddsOptions.RandnFactor;

useFixedHyperparameters  = shapeoddsOptions.useFixedHyperparameters;
useOneLambdaForAll       = shapeoddsOptions.useOneLambdaForAll;

%% label maps characteristics

% observed space dimensionality, number of samples and individual sample
% dimensions
[D, N, fdims] = GetObservedSamplesCharacteristics(fs);

if ~isempty(fsTest)
    [D_test, N_test, fdims_test] = GetObservedSamplesCharacteristics(fsTest);
    
    if (D_test ~= D) || (sum(abs(fdims - fdims_test)) > 0)
        halt_msg = 'training and testing label maps do not have matching dimensions';
        fprintf('%s ... \n', halt_msg);
        return
    end
end

%% vectorize all the label maps for parameter estimation
fs = VectorizeMaps(fs, D, N);
if ~isempty(fsTest)
    fsTest = VectorizeMaps(fsTest, D, N_test);
end

%% optimization options structures
[estep_options, mstep_options] = GetEMOptimizationOptions(L, D, fdims, priorType, ...
    EOptimizationMethod, MOptimizationMethod, isFA, ispar, nworkers, TolFun, TolX, mstep_display);

%% compute the parameters for the piecewise quadratic bound
alpha = ComputePiecewiseBoundParameters();

%% model initialization
gamma = [];
Epsi  = [];
Epsi.betas   = InitBeta   .* ones(1,     L); % ARD prior parameters
Epsi.lambdas = InitLambda .* ones(1, 1 + L); % smoothness prior parameters

if strcmp(shapeoddsOptions.W_init_method, 'matrix') && strcmp(shapeoddsOptions.w0_init_method, 'vector')
    
    [Theta, ~]  = InitializeTheta(fs, D, shapeoddsOptions);
    
    % H-Step: hyperparameters optimization using the current model settings
    if ~useFixedHyperparameters
        Epsi  = Hstep(Theta, fdims, useOneLambdaForAll);
    end
    
elseif strcmp(shapeoddsOptions.W_init_method, 'randn') || strcmp(shapeoddsOptions.W_init_method, 'rand')
    
    logPosterior = []; loglikelihood = [];
    Espi_store   = {};
    Theta_store  = {};
    gamma_store  = {};
    
    for trial = 1 : NumberOfInits
        
        [Theta, ~]  = InitializeTheta(fs, D, shapeoddsOptions);
        
        % better initialization for avoiding wiggles in the loading matrix W
        if strcmp(shapeoddsOptions.W_init_method, 'randn')
            Theta.W             = Theta.W .* RandnFactor;
        end
        
        % E-Step (i.e. inference step)
        if strcmp(shapeoddsOptions.W_init_method, 'randn')
            fprintf('Performing E-step: random trial = %d of %d, randn factor = %f ...\n', trial, NumberOfInits, RandnFactor);
        else
            fprintf('Performing E-step: random trial = %d of %d ...\n', trial, NumberOfInits);
        end
        [factorsPosterior, gamma, loglikelihood(trial)] = Estep(fs, Theta, alpha, estep_options, []);
        
        % M-step: optimize for model parameters
        posteriorMean = factorsPosterior.posteriorMean;
        posteriorCov  = factorsPosterior.posteriorCov;
        
        if strcmp(shapeoddsOptions.W_init_method, 'randn')
            fprintf('Performing M-step: random trial = %d of %d, randn factor = %f ...\n', trial, NumberOfInits, RandnFactor);
        else
            fprintf('Performing M-step: random trial = %d of %d ...\n', trial, NumberOfInits);
        end
        mstep_options.useARD               = useARD;
        mstep_options.useSmoothness        = useSmoothness;
        mstep_options.usePriorForObjective = usePriorForObjectiveInInit; % if enabled useful for initialization since the objective surface now is even rougher
        mstep_options.PriorFactor          = InitPriorFactor;
        
        % optimize for theta
        [Theta, curlogPosterior, optlog]   = Mstep(fs, Theta, Epsi, posteriorMean, posteriorCov, gamma, alpha, mstep_options);
                
        % H-Step: hyperparameters optimization using the current model settings
        if ~useFixedHyperparameters
            Epsi  = Hstep(Theta, fdims, useOneLambdaForAll);
        end
        
        logPosterior(trial)  = sum(curlogPosterior);
        Espi_store{trial}    = Epsi;
        Theta_store{trial}   = Theta;
        gamma_store{trial}   = gamma;
    end
    
    [~, winner] = max(logPosterior);
    if ~isempty(winner)
        Epsi        = Espi_store{winner};
        Theta       = Theta_store{winner};
        gamma       = gamma_store{winner};
    end
    
    clear Espi_store Theta_store gamma_store 
end

%% start alternating between E, M and H steps
logs       = [];
logs.estep = [];
logs.hstep = [];
logs.mstep = [];
logs.trainLoglikelihoodBound = [];
logs.trainLoglikelihood = [];

if ~isempty(fsTest)
    gammaTest = [];
end

if saveEMIters
    save([out_prefix 'em_init.mat'], 'Theta','Epsi','alpha', 'estep_options', 'mstep_options', '-v7.3');
end

%%
loglikelihood = inf;
for iter = 1 : maxIter
    L
    %% keep track of before-iteration values for convergence test
    loglikelihood_old = loglikelihood;
    Theta_old         = Theta;
    
    %% E-Step (i.e. inference step)
    % for each label map, optimize for its global variational parameters
    % gamma_n = {m_n, V_n}
    
    fprintf('Performing E-step: iter = %d of %d ...\n', iter, maxIter);
    estep_options.updateFuncParams.L = L; % in case latent dimensionality changed during prunning out
    [factorsPosterior, gamma, loglikelihood] = Estep(fs, Theta, alpha, estep_options, gamma);
    
    logs.estep(iter).optlog                = factorsPosterior.optlog_store;
    logs.estep(iter).loglikelihood         = loglikelihood;
    logs.estep(iter).factorsPosterior      = factorsPosterior;
    
    logs.trainLoglikelihoodBound(iter)      = loglikelihood;
    [trainLoglikelihood, phisTrain, zTrain] = EvaluateShapeOdds(Theta, factorsPosterior, fs);
    
    logs.trainLoglikelihood(iter) = mean(trainLoglikelihood);
    
    %% M-step: optimize for model parameters
    posteriorMean = factorsPosterior.posteriorMean;
    posteriorCov  = factorsPosterior.posteriorCov;
    
    fprintf('Performing M-step: iter = %d of %d ...\n', iter, maxIter);
    % update the mstep update function parameters using the current
    % optimized hyperparameters setting
    mstep_options.updateFuncParams.L   = L; % in case latent dimensionality changed during prunning out
    mstep_options.useARD               = useARD;
    mstep_options.useSmoothness        = useSmoothness;
    
    mstep_options.usePriorForObjective = usePriorForObjectiveInOptimization; % if enabled, giving it a chance to refine over initialization on a smoother energy surface
    % if turn it off, since each component already claims his subspace, having it off might (?) make beta and lamdba estimation stable
    
    % decaying prior factor
    if usePriorForObjectiveInOptimization
        lambda_0_term = (RegularizationContractiveFactor^iter) * StartingRegularization;
        error_term    = ((1-(RegularizationContractiveFactor^iter))/(eps+1-RegularizationContractiveFactor))*EndingRegularization;
        lambda_t      =  lambda_0_term + error_term;
        
        mstep_options.PriorFactor = lambda_t;
    else
        mstep_options.PriorFactor = 0;
    end
    
    % optimize for theta
    [Theta, logPosterior, optlog]   = Mstep(fs, Theta, Epsi, posteriorMean, posteriorCov, gamma, alpha, mstep_options);
            
    logs.mstep(iter).logPosterior = sum(logPosterior);
    logs.mstep(iter).optlog       = optlog;
    
    %% H-Step: hyperparameters optimization using the current model settings
    if ~useFixedHyperparameters
        Epsi                  = Hstep(Theta, fdims, useOneLambdaForAll);
    end
    logs.hstep(iter).Epsi = Epsi;
    
    Epsi.lambdas
    Epsi.betas
    
    %%
    if ~isempty(fsTest)
        [factorsPosteriorTest, gammaTest, loglikelihood] = Estep(fsTest, Theta, alpha, estep_options, gammaTest);
        
        [testLoglikelihood, phisTest, zTest] = EvaluateShapeOdds(Theta, factorsPosteriorTest, fsTest);
        
        logs.testLoglikelihood(iter)         = mean(testLoglikelihood);
        logs.testLoglikelihoodBound(iter)    = loglikelihood;
        logs.factorsPosteriorTest(iter)      = factorsPosteriorTest;
    end
    
    %% save current iteration for debugging
    if saveEMIters
        save([out_prefix 'em_iter' num2str(iter) '.mat'], 'Theta','Epsi','alpha', 'estep_options', 'mstep_options', '-v7.3');
    end
    
    %% check for convergence
    dW = []; dw0 = [];
    dW    = max(max(abs(Theta_old.W  - Theta.W)  / (sqrt(eps)+max(max(abs(Theta.W))))));
    dw0   = max(max(abs(Theta_old.w0 - Theta.w0) / (sqrt(eps)+max(max(abs(Theta.w0))))));
    delta = max(dW, dw0);
    
    %% prune out irrelevant latent dimensionality
    % TODO: death moves in case of components do not have enough data
    % support (according to their responsibilities) or enough latent
    % dimensionality ... a component ended up to have all its loadings
    % irrelevant
    if useARD && PruneIrrelevantLatentDimensions
        [Theta, Epsi, gamma, L, loadingIdx] = PruneOutIrrelevantLatentDimensions(Theta, Epsi, gamma, L, LoadingVarianceThreshold);
    end
    
    %%
    fprintf('delta = %f, TolX = %f, abs(loglikelihood - loglikelihood_old) = %f, TolFun = %f ... \n',delta, TolX, abs(loglikelihood - loglikelihood_old), TolFun);
    %if iter > 5
        if delta < TolX
            break;
        elseif abs(loglikelihood - loglikelihood_old)  < TolFun
            break;
        end
    %end
    %%
end

%% E-Step (i.e. inference step) to reflect the last M-step before "convergence"
% for each label map, optimize for its global variational parameters
% gamma_n = {m_n, V_n}

fprintf('Performing the final E-step ...\n');
[factorsPosterior, gamma, loglikelihood] = Estep(fs, Theta, alpha, estep_options, gamma);

posteriorMean = factorsPosterior.posteriorMean;
posteriorCov  = factorsPosterior.posteriorCov;

Z                                     = factorsPosterior.posteriorMean;
logs.estep(end+1).optlog              = factorsPosterior.optlog_store;
logs.estep(end).loglikelihood         = loglikelihood;
logs.estep(end).factorsPosterior      = factorsPosterior;
logs.trainLoglikelihoodBound(end+1)   = loglikelihood;

[trainLoglikelihood, phisTrain, zTrain] = EvaluateShapeOdds(Theta, factorsPosterior, fs);
logs.trainLoglikelihood(end+1)          = mean(trainLoglikelihood);

if ~isempty(fsTest)
    
    [factorsPosteriorTest, gammaTest, loglikelihood] = Estep(fsTest, Theta, alpha, estep_options, gammaTest);
    
    [testLoglikelihood, phisTest, zTest] = EvaluateShapeOdds(Theta, factorsPosteriorTest, fsTest);
    
    logs.testLoglikelihood(end+1)         = mean(testLoglikelihood);
    logs.testLoglikelihoodBound(end+1)    = loglikelihood;
    logs.factorsPosteriorTest(end+1)      = factorsPosteriorTest;
end

%% H-Step: hyperparameters optimization using the current model settings
if ~useFixedHyperparameters
    Epsi                      = Hstep(Theta, fdims, useOneLambdaForAll);
end
logs.hstep(end+1).Epsi = Epsi;

%% save the final debugging info
if saveEMFinal
    save([out_prefix 'em_final.mat'], 'Theta','Epsi','alpha', 'estep_options', 'mstep_options', '-v7.3');
end

%% report "convergence" ...
if iter < maxIter
    fprintf('EM converged with %d iterations ...\n' , iter);
else
    fprintf('EM did not converged with max %d iterations ...\n' , maxIter);
end

%% result structure
if strcmp(shapeoddsOptions.W_init_method, 'randn')
    resStruct.RandnFactor      = RandnFactor;
end

resStruct.D      = D;
resStruct.fdims  = fdims;

resStruct.Theta              = Theta;
resStruct.Espi               = Epsi;
resStruct.factorsPosteriorTrain    = factorsPosterior;

if ~isempty(fsTest)
    resStruct.factorsPosteriorTest     = logs.factorsPosteriorTest(end);
end

resStruct.gamma = gamma;
resStruct.alpha = alpha;
resStruct.logs   = logs;

resStruct.estep_options = estep_options;
resStruct.mstep_options = mstep_options;
resStruct.shapeoddsOptions   = shapeoddsOptions;

resStruct.trainLogLikelihood   = logs.trainLoglikelihood(end);
if ~isempty(fsTest)
    resStruct.testLogLikelihood    = logs.testLoglikelihood(end);
end

resStruct.trainLogLikelihoodBound  = logs.trainLoglikelihoodBound(end);
if ~isempty(fsTest)
    resStruct.testLogLikelihoodBound   = logs.testLoglikelihoodBound(end);
end