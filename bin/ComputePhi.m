function phi = ComputePhi(q, epsilon)

if ~exist('epsilon')
    epsilon = 1e-50;
end

q    = BoundQ(q, epsilon);
phi  = real(log(epsilon+(q./(1-q+epsilon))));