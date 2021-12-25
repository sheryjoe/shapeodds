
function q = BoundQ(q, epsilon)

if ~exist('epsilon')
    epsilon = 1e-50;
end

q = double(q);
q(q>=(1 - epsilon)) = 1 - epsilon;
q(q<=epsilon)       = epsilon;