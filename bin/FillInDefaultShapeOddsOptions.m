
function options = FillInDefaultShapeOddsOptions(N)


options = [];

% latent space dimensionality
options.L = N-1;

% number of EM iterations
options.maxIter = 100; %50; %100;

% type of prior
options.priorType   = 'laplacian-square';

% whether the model to be build is a factor analysis model with standard normal distribution for the latent variable
options.isFA   = 0;

% the initialization method for offset vector
options.w0_init_method = 'randn'; 
options.w0_init        = [];
            
% the initialization method for loading vectors
options.W_init_method = 'randn';
options.W_init        = [];

% small value to bound label maps
options.epsilon = 1e-100; % very tiny values will cause staircase artifacts when going to log-odds space

% visualization flag
options.show = 0;

% verbose flag
options.verbose = 0;

% a flag whether to save each inidividual result for EM, note that you
% might run out of storage (only used for debugging purposes)
options.saveEMIters = 1;
options.saveEMFinal = 1;
options.out_prefix  = './models/'; % in case any of the previous save flag are set to 1

% optimization method to be used for Estep
options.EOptimizationMethod = 'lbfgs';

% optimization method to be used for Mstep
options.MOptimizationMethod = 'cg';

% use this to associate/evaluate a testing set when we perform model
% training
options.fsTest = [];

% display level for Mstep
options.mstep_display = 'iter';

% tolerance to be used for optimization (E and M step)
options.TolX          = 1e-6;
options.TolFun        = 1e-6;

% use parfor for estep
options.ispar         = 1;
options.nworkers      = 4;

% flag to enable automatic relevance selection prior for latent
% dimensionality pruning
options.useARD = 1;
options.useSmoothness           = 1;

% number of randn inits to improve initialization
options.NumberOfInits = 5;

% initial smoothness and ARD priors
options.InitBeta   = 1e-5;
options.InitLambda = 1;

% for prunning out irrelevant loading vectors
options.LoadingVarianceThreshold        = 1e-16; 
options.PruneIrrelevantLatentDimensions = 1;

% enable prior to affect time step computation
options.usePriorForObjectiveInInit         = 1;
options.usePriorForObjectiveInOptimization = 1;
options.InitPriorFactor                    = 1;

% prior weight will be 1 for all optimization iterations
options.StartingRegularization          = 1;
options.EndingRegularization            = 1; % will end at  [1/(1-factor)] * EndingReguarlization 
options.RegularizationContractiveFactor = 1; % (0,1): higher means slower decay

options.RandnFactor              = 0.05;

options.useFixedHyperparameters = 0;
options.useOneLambdaForAll      = 0;
