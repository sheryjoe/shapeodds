
function [neglogLikelihood, neglogLikelihoodGradient] = SampleNegLogLikelihoodBound(gamma_n, Theta, f_n, alpha, computeGradient)

verbose = 0;

% Theta = {Mu, Sigma/SigmaInv, W, w0}: model parameters
% f_n: the nth label map
% gamma_n = {m_n, V_n}: global variational parameters for the nth sample
% alpha : local variational bound parameters (piecewise quadratic bound)

if verbose
    tic
end

% model parameters
Mu          = Theta.Mu;
invSigma    = Theta.invSigma;
logdetSigma = Theta.logdetSigma;

% latent space dimensionality
L   = size(Mu,1);

% the loading matrix (note that it is in a vector form)
W = Theta.W;

% the offset vector
w0 = Theta.w0;

% extract current global variational parameters
[m_n, V_n] = UnpackVariationalParams(gamma_n, L);

% initialize
% the lower bound of the loglikelihood of the nth sample (L_n)
logLikelihood       = 0; 
if computeGradient
    gradientWrtMean     = zeros(L,1); % dL_n/dm_n in the latent space
    gradientWrtVariance = zeros(L,L); % dL_n/dV_n in the latent space
else
    neglogLikelihoodGradient = 0;
end


% check if current covariance matrix is positive definite
% get the upper triangle of positive definite of the current covariance matrix, this
% is very useful when we have roundoff errors, if the current matrix is not
% positive definite (up to roundoff errors) we need to exit the
% optimization
[V_n, fixed, U_n] = FixCovarianceMatrix(V_n); 
if ~fixed 
    neglogLikelihood         = -inf;
    neglogLikelihoodGradient = 0;
    return
end

% compute determinant and inverse of the sample's covariance matrix
try
    logdetV_n = logdetSPD(V_n); % special implementation to avoid overflow errors
catch exception
    logdetV_n = log(det(V_n) + eps);
end
invV_n    = solve_chol(U_n, eye(L)); %U_n\(U_n'\eye(L));

% contribution from the KL divergence term
MMu               = m_n - Mu;
invSigmaMMu       = invSigma * MMu;
trace_Vn_invSigma = V_n(:)'*invSigma(:);
negKL_term        = 0.5*(logdetV_n - logdetSigma - trace_Vn_invSigma - MMu'* invSigmaMMu + L);

logLikelihood       = logLikelihood       + negKL_term;
if computeGradient
    gradientWrtMean     = gradientWrtMean     -  invSigmaMMu;
    gradientWrtVariance = gradientWrtVariance +  0.5*(invV_n - invSigma);
end

% contribution from the loglikelihood expectation term 
% mtilde and vtilde (change of variables from the latent space to the
% natural parameters space)
m_tilde_n = W*m_n + w0;
V_tilde_n = sum(W'.*(V_n*W'),1)'; % V_tilde_n = diag(W*V_n*W');

% contribution from the tractable term of the loglikelihood expectation
logLikelihood   = logLikelihood   +  f_n' * m_tilde_n;
if computeGradient
    gradientWrtMean = gradientWrtMean +  W' * f_n;
end

% contribution from the upper bound of the intractable term of the loglikelihood expectation 
% the upper bound term along with its derviatives with respect to mean and
% variance in the natural parameters space
[upper_bound_term, g_m_n_tilde, G_V_n_tilde]  = funObj_pw(m_tilde_n, V_tilde_n, alpha);

logLikelihood           = logLikelihood       - sum(upper_bound_term);
if computeGradient
    gradientWrtMean     = gradientWrtMean     - W' * g_m_n_tilde;
    gradientWrtVariance = gradientWrtVariance - W' * bsxfun(@times, G_V_n_tilde(:), W); %gV = gV - W'*diag(gvb)*W;
    
    % add double contribution for off-diagonal elements since we are only
    % maintaing the upper triangle elements during the optimization (the actual
    % degree of freedom for a covariance matrix)
    gradientWrtVariance = gradientWrtVariance + triu(gradientWrtVariance,1); % exclude the diagonal terms
    
    % collect gradients
    logLikelihoodGradient    = [gradientWrtMean(:);
                                gradientWrtVariance(triu(ones(L))==1)]; % only retain the upper triangle
end

% return negative the function and negative its gradient for minimization
neglogLikelihood         = -logLikelihood;
if computeGradient
    neglogLikelihoodGradient = -logLikelihoodGradient;
end

if verbose
    elapsedTime = toc;
    fprintf('Estep: SampleNegLogLikelihoodBound took %f seconds ...\n', elapsedTime);
end
