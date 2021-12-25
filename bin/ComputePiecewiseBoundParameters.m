
function alpha = ComputePiecewiseBoundParameters()

% compute the parameters for the piecewise quadratic bound
% this local bound parameters are fixed along the variational learning
% process since they are independent of the model parameters and the global
% variational parameters

% number of pieces
R = 20; % max number allowed

% bound type
boundType = 'quad';

% bound parameters (alpha), each piece is defined by l = t_r-1, h = t_r and
% a = [a_r, b_r, c_r]
alpha = getPiecewiseBound(boundType, R);