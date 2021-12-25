
function [V_n, fixed, U_n] = FixCovarianceMatrix(V_n)

verbose = 0;
fixed   = 1;

[U_n,pp] = chol(V_n);
posdef   = (pp == 0);

if ~posdef
    
    if verbose
        fprintf('Converting V_n to a postive definite matrix ... ');
    end
    
    if all(isnan(V_n(:)))
        tt=0;
    end
    V_n = nearestSPD(V_n);
    
    % Ahat failed the chol test. It must have been just a hair off,
    % due to floating point trash, so it is simplest now just to
    % tweak by adding a tiny multiple of an identity matrix.
    
    %[eV,eD] = eig(V_n);
    
    % We can choose what should be a reasonable rank 1 update to V_n that will make it positive definite.
    % Any more of a perturbation in that direction, and it would truly be positive definite.
    
    %eV1  = eV(:,1);
    IdMatrix = eye(size(V_n));
    for eta = linspace(0,1000,100000)
        %V_n_ = V_n + eV1*eV1'*eta*(eps(eD(1,1))-eD(1,1));
        V_n_ = V_n + eta .* IdMatrix;
        
        [U_n,pp] = chol(V_n_);
        posdef   = (pp == 0);
    
        if posdef
            break
        end
    end
    
    if verbose
        fprintf('eta = %f, posdef = %d ...\n', eta, posdef);
    end
    
    if ~posdef
        fixed = 0;
        fprintf('WARNING: Covariance matrix is not fixed !!!! ...\n');
    else
        fixed = 1;
        V_n   = V_n_;
    end
end