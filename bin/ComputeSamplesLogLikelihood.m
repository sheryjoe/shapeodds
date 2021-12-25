function loglikelihoods = ComputeSamplesLogLikelihood(fs, phis, useQ)

[D, N] = size(fs);

if ~exist('epsilon', 'var')
    epsilon = 1e-30;
end

if ~exist('useQ', 'var')
    useQ = 0;
end

loglikelihoods = zeros(1,N);

for n = 1 : N
    f_n   = fs(:,n);
    phi_n = phis(:,n);
    
    if ~useQ
        loglikelihoods(n) = sum(f_n.*phi_n - log(1 + exp(phi_n)));
    else
        q_n = ComputeQ(phi_n);
        
        % to limit q \in (0,1)
        q_n(find(q_n==1)) = 1 - epsilon;
        q_n(find(q_n==0)) = epsilon;
        
        likelihood_map = f_n.*log(q_n + epsilon) + (1-f_n).*log(1-q_n + epsilon);
        loglikelihoods(n) = sum(likelihood_map);
    end
end
