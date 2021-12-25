function [m_n, V_n] = UnpackVariationalParams(gamma_n, L)

% extract current global variational parameters
m_n = gamma_n(1:L); % the first L-terms are the elements of the mean
m_n = m_n(:);
idx = L + [1:L*(L+1)/2]; % the rest of the terms are the upper triangle elements of the covariance
V_n = unpackcovariance(gamma_n(idx),L);