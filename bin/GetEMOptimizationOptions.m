
function [estep_options, mstep_options] = GetEMOptimizationOptions(L, D, fdims, priorType, EOptimizationMethod, MOptimizationMethod, ...
    isFA, ispar, nworkers, TolFun, TolX, mstep_display)

%% optimization options structure for the Estep
estep_options             = [];
estep_options.display     = 'final' ; %'iter'; %'excessive'; %'full';
estep_options.Method      = EOptimizationMethod;
estep_options.maxFunEvals = 1000;%3000;
estep_options.MaxIter     = 1000;%2000;
estep_options.optTol      = 1e-6;  % Termination tolerance on the first-order optimality
estep_options.progTol     = 1e-8; % Termination tolerance on progress in terms of function/parameter changes

% estep_options.Fref        = 1;  % controls non-monotonicity, 1 for lbfgs
estep_options.numDiff     = 0;  % compute derivatives using user-supplied function
estep_options.DerivativeCheck = 'off' ; % computes derivatives numerically at initial point and compares to user-supplied derivative

estep_options.TolFun      = TolFun;
estep_options.TolX        = TolX;

estep_options.ispar        = ispar;
estep_options.nworkers     = nworkers;

% the update function
estep_options.updateFuncObj      = @UpdateVariationalParams;
estep_options.updateFuncParams   = [];
estep_options.updateFuncParams.L = L;


%% optimization options structure for the Mstep
mstep_options             = [];
mstep_options.display     = mstep_display; %'iter'; %'excessive'; %'full';
mstep_options.Method      = MOptimizationMethod;
mstep_options.maxFunEvals = 500;%1000;
mstep_options.MaxIter     = 500;%1000;
mstep_options.optTol      = 1e-6;  % Termination tolerance on the first-order optimality
mstep_options.progTol     = 1e-8; % Termination tolerance on progress in terms of function/parameter changes

% mstep_options.Fref        = 1;  % controls non-monotonicity, 1 for lbfgs
mstep_options.numDiff     = 0;  % compute derivatives using user-supplied function
mstep_options.DerivativeCheck = 'off' ; % computes derivatives numerically at initial point and compares to user-supplied derivative

mstep_options.TolFun      = TolFun;
mstep_options.TolX        = TolX;

% the update function
mstep_options.updateFuncObj                 = @UpdateModelParams;
mstep_options.updateFuncParams              = [];
mstep_options.updateFuncParams.priorType    = priorType;
mstep_options.updateFuncParams.fdims        = fdims; % to be able to unvectorize the vectorized maps when applying the implicit update scheme
mstep_options.updateFuncParams.L    = L;
mstep_options.updateFuncParams.D    = D;
mstep_options.updateFuncParams.ispar    = ispar;
mstep_options.updateFuncParams.nworkers = nworkers;

% type of the latent gaussian model
mstep_options.isFA = isFA;
