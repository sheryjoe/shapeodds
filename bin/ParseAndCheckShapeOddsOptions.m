
function [options, halt, halt_msg] = ParseAndCheckShapeOddsOptions(options)

halt     = 0;
halt_msg = '';

% latent space dimensionality
if ~isfield(options, 'L')
    halt_msg = 'Latent space dimensionality (L) is needed be to specified (max value: N - 1)';
    halt     = 1;
    return;
end

% number of EM iterations
if ~isfield(options, 'maxIter')
    maxIter = 1000;
    options.maxIter = maxIter;
end

% type of prior
if ~isfield(options, 'priorType')
    options.priorType   = 'laplacian-square'; % 'dirichlet', 'laplacian-square'
end

% whether the model to be build is a factor analysis model with standard normal distribution for the latent variable
if ~isfield(options, 'isFA')
    options.isFA   = 0;
end

% the initialization method {'rand','zeros','voting','avg_sdm', 'vector'}
if ~isfield(options, 'w0_init_method')
    w0_init_method         = 'randn';
    options.w0_init_method = w0_init_method;
else
    if strcmp(options.w0_init_method, 'vector')
        if ~isfield(options, 'w0_init')
            halt_msg = 'vector initialization for the offset vector (w0) is specified without w0_init vector ...!!!';
            halt     = 1;
            return;
        end    
    end
    
end

if ~isfield(options, 'W_init_method')
    W_init_method         = 'randn';
    options.W_init_method = W_init_method;
else
    if strcmp(options.W_init_method, 'matrix')
        if ~isfield(options, 'W_init')
            halt_msg = 'matrix initialization for loading vectors (W) is specified without W_init matrix ...!!!';
            halt     = 1;
            return;
        end    
    end
        
end

% small value to bound label maps
if ~isfield(options, 'epsilon')    
    epsilon = 1e-100;
    options.epsilon = epsilon;
end

% visualization flag
if ~isfield(options, 'show')
    options.show = 0;
end

% verbose flag
if ~isfield(options, 'verbose')
    options.verbose = 0;
end

if ~isfield(options, 'saveEMIters')
    options.saveEMIters = 0;
end

if ~isfield(options, 'saveEMFinal')
    options.saveEMFinal = 0;
end

if ~isfield(options, 'out_prefix')
    if options.saveEMIters || options.saveEMFinal
        halt_msg = 'either saveEMIters or saveEMFinal was enabled without providing an output_prefix ...!!!';
        halt     = 1;
        return;
    else
        options.out_prefix = './models/';
    end
end


if ~isfield(options, 'EOptimizationMethod')
    options.EOptimizationMethod = 'lbfgs';
end

if ~isfield(options, 'MOptimizationMethod')
    options.MOptimizationMethod = 'cg';
end

if ~isfield(options, 'fsTest')
    options.fsTest = [];
end

if ~isfield(options, 'mstep_display')
    options.mstep_display = 'iter';
end

% tolerance to be used for optimization (E and M step)
if ~isfield(options, 'TolX')
    options.TolX          = 1e-6;
end

if ~isfield(options, 'TolFun')
    options.TolFun          = 1e-6;
end

if ~isfield(options, 'useARD')
    options.useARD          = 1;
end

if ~isfield(options, 'NumberOfInits')
    options.NumberOfInits = 5;
end


% for prunning out irrelevant loading vectors, didn't work
if ~isfield(options, 'LoadingVarianceThreshold')
    options.LoadingVarianceThreshold = 1e-5; 
end

% initial smoothing and ard priors
if ~isfield(options, 'InitBeta')
    options.InitBeta = 1;
end
if ~isfield(options, 'InitLambda')
    options.InitLambda = 1;
end

% enable prior to affect time step computation
if ~isfield(options, 'usePriorForObjectiveInInit')
    options.usePriorForObjectiveInInit = 1;
end
if ~isfield(options, 'usePriorForObjectiveInOptimization')
    options.usePriorForObjectiveInOptimization = 1;
end
if ~isfield(options, 'InitPriorFactor')
    options.InitPriorFactor = 1;
end

if ~isfield(options, 'RandnFactor')
    options.RandnFactor = 0.05;
end


