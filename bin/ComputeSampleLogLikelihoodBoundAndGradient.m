
function [logLikelihood, gw0, gW] = ComputeSampleLogLikelihoodBoundAndGradient(f_n, gamma_n, alpha, W, w0, computeGradient, logLikelihood, gw0, gW)

% latent dimensionality
L = size(W, 2);

% extract current global variational parameters
[m_n, V_n]          = UnpackVariationalParams(gamma_n, L);

% mtilde and vtilde change of variables from the latent space to the
% natural parameters space)
m_tilde_n = W*m_n + w0;
V_tilde_n = sum(W'.*(V_n*W'),1)'; % V_tilde_n = diag(W*V_n*W');

% first term of the second term of the likelihood
logLikelihood = logLikelihood + f_n' * m_tilde_n;

% upper bound of the second term and its gradients
% the upper bound term along with its derviatives with respect to mean and
% variance with respect to mean and
% variance in the natural parameters space
[llp_upper_bound, g_m_n_tilde, G_V_n_tilde]  = funObj_pw(m_tilde_n, V_tilde_n, alpha);

logLikelihood    = logLikelihood + sum(-llp_upper_bound);

if computeGradient
    % gradient wrt w0
    gw0 = f_n - g_m_n_tilde;
    
    % gradient wrt W
    gW  = f_n * m_n' - g_m_n_tilde * m_n';
    
    for l = 1 : L
        gW(:,l)  = gW(:,l) - 2 * G_V_n_tilde(:) .* (W * V_n(:,l));
    end
end