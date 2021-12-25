function gamma_n = PackVariationalParams(m_n, V_n)

% latent space dimensionality
L = size(m_n,1);

% number of actual degrees of freedom = L (for mean) plus the number of
% elements in the upper triangular matrix of the covariance matrix
nVariationalParams = length(find(triu(ones(L)))) + L;

gamma_n          = zeros(nVariationalParams,1);
gamma_n(1:L)     = m_n;
gamma_n(L+1:end) = packcovariance(V_n);