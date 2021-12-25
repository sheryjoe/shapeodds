
function [Theta, logPosterior, optlog] = Mstep(fs, Theta, Epsi, posteriorMean, posteriorCov, gamma, alpha, options)

% optimize for the model paramters Theta = {Mu, Sigma, W, w0}

% observed space dimensionality and  number of samples
% note that label maps were already vectorized
[D, N] = size(fs);

% latent space dimensionality
L = size(Theta.Mu,1);

% now update model paramters

% first update the latent variable prior parameter (only takes effect
% if the model is not FA
if options.isFA == 0
    Theta = UpdateLatentPriorParameters(Theta, posteriorMean, posteriorCov);
end

% prepare the initial offset vector and loading matrix which are the
% current ones
Wtilde = [Theta.w0 Theta.W];
Wvec   = Wtilde(:);

computeGradient = 1;
options.updateFuncParams.L            = L;
options.updateFuncParams.Epsi         = Epsi;
options.updateFuncParams.useSmoothness = options.useSmoothness;

[Wvec, neglogPosterior, ~, optlog] = minFunc2(@NegLogPosteriorBound, Wvec, options, Theta, fs, gamma, alpha, ...
                                                    options.updateFuncParams.priorType, Epsi, ...
                                                    options.updateFuncParams.fdims, options.useARD, options.useSmoothness,...
                                                    options.usePriorForObjective, options.PriorFactor, computeGradient, ...
                                                    options.updateFuncParams.ispar, options.updateFuncParams.nworkers);

Wtilde = reshape(Wvec, [D, 1 + L]);

% update model parameters with the current loading matrix and offset vector
Theta.w0 = Wtilde(:,1);
Theta.W  = Wtilde(:,2:end);

logPosterior = -neglogPosterior;

