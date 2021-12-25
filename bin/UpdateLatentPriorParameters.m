
function [Theta] = UpdateLatentPriorParameters(Theta, posteriorMean, posteriorCov)

% update the latent variable distribution parameters (Mu and Sigma), these
% updates are in closed form solution
%MMu      = bsxfun(@minus, posteriorMean , Theta.Mu);
%V_n      = bsxfun(@plus,  posteriorCov, MMu*MMu');

[L,N] = size(posteriorMean);
V_n   = zeros(L,L,N);
for n = 1 : N
    MMu        = posteriorMean(:,n)  - Theta.Mu;
    V_n(:,:,n) = posteriorCov(:,:,n) + MMu*MMu';
end

Theta.Mu          = mean(posteriorMean,2);
Theta.Sigma       = mean(V_n, 3);
Theta.Sigma       = FixCovarianceMatrix(Theta.Sigma);
Theta.invSigma    = inv(Theta.Sigma);
Theta.logdetSigma = logdetSPD(Theta.Sigma);

