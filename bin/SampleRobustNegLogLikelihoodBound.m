
function [E_R, gradE_R, outlier_mask, E_gamma_x] = SampleRobustNegLogLikelihoodBound(gamma_n, Theta, f_n, alpha, tau, funcType, computeGradient)

% if ~exist('verbose', 'var')
verbose = 0;
% end

% Theta = {Mu, Sigma/SigmaInv, W, w0}: model parameters
% f_n: the nth label map
% gamma_n = {m_n, V_n}: global variational parameters for the nth sample
% alpha : local variational bound parameters (piecewise quadratic bound)

if verbose
    tic
end

% model parameters
Mu     = Theta.Mu;
Sigma  = Theta.Sigma;

% latent space dimensionality
L   = size(Mu,1);

% data space dimensionality
D   = size(f_n,1);

% the loading matrix (note that it is in a vector form)
W = Theta.W;

% the offset vector
w0 = Theta.w0;

% extract current global variational parameters
[m_n, V_n] = UnpackVariationalParams(gamma_n, L);

% prior parameters in the natural parameters space
Mu_tilde    = W*Mu + w0;
Sigma_tilde = sum(W'.*(Sigma*W'),1)'; 

% latent posterior variational parameters mtilde and vtilde (change of variables 
% from the latent space to the natural parameters space)
m_tilde_n = W*m_n + w0;
V_tilde_n = sum(W'.*(V_n*W'),1)'; % V_tilde_n = diag(W*V_n*W');

% initialize
% the objective function to be robustified (defined pixel-wise)
E_gamma_x = zeros(D,1);
if computeGradient
    gradE_gamma_x_m    = zeros(L,D); 
    gradE_gamma_x_V    = zeros(L,L,D); 
else
    gradE_R = 0;
end

% contribution from the KL divergence term
logdetV_n_x         = log(abs(V_tilde_n)   + eps);
logdetSigma_x       = log(abs(Sigma_tilde) + eps);
trace_Vn_invSigma_x = V_tilde_n ./ (Sigma_tilde + eps);
MMu2_x              = ((m_tilde_n - Mu_tilde).^2)./ (Sigma_tilde + eps);
negKL_x             = 0.5*(logdetV_n_x - logdetSigma_x - trace_Vn_invSigma_x - MMu2_x + 1);

% the objective function to be robustified (defined pixel-wise)
E_gamma_x           = E_gamma_x - negKL_x;

if computeGradient
    MMu_x           = (m_tilde_n - Mu_tilde)./ (Sigma_tilde + eps);
    VSigma_x        = -0.5 .* ( (1./(V_tilde_n + eps)) - (1./(Sigma_tilde + eps)));
    
    for d = 1 : D
        gradE_gamma_x_m(:,d)   = gradE_gamma_x_m(:,d)   + MMu_x(d) .* W(d,:)';
        gradE_gamma_x_V(:,:,d) = gradE_gamma_x_V(:,:,d) + VSigma_x(d) * W(d,:)' *  W(d,:);
    end
end

% contribution from the loglikelihood expectation term 
% contribution from the tractable term of the loglikelihood expectation
E_gamma_x = E_gamma_x - f_n .* m_tilde_n;
if computeGradient
    gradE_gamma_x_m = gradE_gamma_x_m - bsxfun(@times, f_n(:), W)';
end

% contribution from the upper bound of the intractable term of the loglikelihood expectation 
% the upper bound term along with its derviatives with respect to mean and
% variance in the natural parameters space
[upper_bound_term, g_m_n_tilde, G_V_n_tilde]  = funObj_pw(m_tilde_n, V_tilde_n, alpha);

E_gamma_x = E_gamma_x + upper_bound_term;

if computeGradient
    for d = 1 : D
        gradE_gamma_x_m(:,d)   = gradE_gamma_x_m(:,d)   + g_m_n_tilde(d) .* W(d,:)';
        gradE_gamma_x_V(:,:,d) = gradE_gamma_x_V(:,:,d) + G_V_n_tilde(d) * W(d,:)' *  W(d,:);
    end
end

% the robustified marginal neg log-likelihood
rho_E_gamma_x = rho_function(E_gamma_x, tau, funcType);
%rho_E_gamma_x = E_gamma_x;
E_R           = sum(rho_E_gamma_x);
outlier_mask  = double(E_gamma_x >= tau);

if computeGradient
    epsi_E_gamma_x = epsi_function(E_gamma_x, tau, funcType);
    %epsi_E_gamma_x = ones(D,1);
       
    for d = 1 : D
        gradE_gamma_x_m(:,d)   = epsi_E_gamma_x(d) .* gradE_gamma_x_m(:,d);
        gradE_gamma_x_V(:,:,d) = epsi_E_gamma_x(d) .* gradE_gamma_x_V(:,:,d);
    end
    
    gradE_gamma_m   = sum(gradE_gamma_x_m,2);
    gradE_gamma_V   = sum(gradE_gamma_x_V,3);
end

if computeGradient

    % add double contribution for off-diagonal elements since we are only
    % maintaing the upper triangle elements during the optimization (the actual
    % degree of freedom for a covariance matrix)
    gradE_gamma_V = gradE_gamma_V + triu(gradE_gamma_V,1); % exclude the diagonal terms
    
    % collect gradients
    gradE_R    = [gradE_gamma_m(:);
                  gradE_gamma_V(triu(ones(L))==1)]; % only retain the upper triangle
end

if verbose
    elapsedTime = toc;
    fprintf('Estep: SampleRobustNegLogLikelihoodBound took %f seconds ...\n', elapsedTime);
end
