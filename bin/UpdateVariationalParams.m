function gamma = UpdateVariationalParams(gamma, timeStep, descentDirection, updateParams)

% apply the update
gamma = gamma + timeStep * descentDirection;

% latent space dimensionality
L  = updateParams.L;

% extract current global variational parameters
[m_n, V_n] = UnpackVariationalParams(gamma, L);

% fix the covariance matrix to be positive definite
verbose      = 0;
[V_n, fixed] = FixCovarianceMatrix(V_n);

% pack-in the updated parameters vector
gamma = PackVariationalParams(m_n, V_n);

