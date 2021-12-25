function logdetA = logdetSPD(A)
% In Bayesian data analysis, the log determinant of symmetric positive definite matrices often pops up as a normalizing constant in MAP estimates with multivariate Gaussians (ie, chapter 27 of Mackay). Oftentimes, the determinant of A will evaluate as infinite in Matlab although the log det is finite, so one can’t use log(det(A)). However, we know that:
% 
%     \mathbf{A} = \mathbf{L}\mathbf{L}' (Cholesky decomposition)
%     |\mathbf{L}| = \prod_i L_{ii} (determinant of a lower triangular matrix)
%     \log \prod_i x_i = \sum_i \log x_i (log rule)
% 
% Thus to calculate the log determinant of a symmetric positive definite matrix:
% note that this method for logdet doesn't suffer from overflow

L       = chol(A);
logdetA = 2*sum(log(diag(L)));