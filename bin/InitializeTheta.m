
function [Theta, shapeoddsOptions] = InitializeTheta(fs, D, shapeoddsOptions)


% offset/bias vector (origin of the natural parameters space)
switch shapeoddsOptions.w0_init_method
    case 'rand'
        if shapeoddsOptions.verbose
            fprintf('Random initialization ...\n');
        end
        w0 = rand(D,1) - 0.5;
        
    case 'randn'
        if shapeoddsOptions.verbose
            fprintf('Random normal initialization ...\n');
        end
        w0 = randn(D,1);
        
    case 'zeros'
        if shapeoddsOptions.verbose
            fprintf('Zeros initialization ...\n');
        end
        w0 = zeros(D,1);
        
    case 'voting'
        if shapeoddsOptions.verbose
            fprintf('Voting-based initialization ...\n');
        end
        
        % we solve for log-odds while the input is a label map
        q0  = mean(fs, 2);
        w0  = ComputePhi(q0, shapeoddsOptions.epsilon);
        
    case 'vector',
        w0 = shapeoddsOptions.w0_init;
    otherwise
        fprintf('Undefined initialization method ... \n');
        return;
end


switch shapeoddsOptions.W_init_method
    case 'rand'
        fprintf('rand initialization method ... \n');
        
        %W = rand(D, shapeoddsOptions.L) - 0.5;
        a = -4 * sqrt(6/(D+shapeoddsOptions.L));
        b =  4 * sqrt(6/(D+shapeoddsOptions.L));
        W = a + (b-a).*rand(D, shapeoddsOptions.L);
        
    case 'randn'
        fprintf('randn initialization method ... \n');
        W = randn(D, shapeoddsOptions.L);
    case 'zeros'
        fprintf('zeros initialization method ... \n');
        W = ones(D,shapeoddsOptions.L).*shapeoddsOptions.epsilon; % to avoid explicit zeros for loading vectors in order to be able to compute the truncated gaussian moments
    case 'ones'
        fprintf('zeros initialization method ... \n');
        W = ones(D,shapeoddsOptions.L);
    case 'matrix',
        fprintf('matrix initialization method ... \n');
        W = shapeoddsOptions.W_init;
        
    otherwise
        fprintf('Undefined initialization method ... \n');
        return;
end

% latent variable distribution parameters
Mu    = zeros(shapeoddsOptions.L,1);
Sigma = eye(shapeoddsOptions.L,shapeoddsOptions.L);

% model parameters structure
Theta             = [];
Theta.Mu          = Mu;
Theta.Sigma       = Sigma;
Theta.invSigma    = inv(Sigma);
Theta.logdetSigma = logdetSPD(Sigma);
Theta.W           = W;
Theta.w0          = w0;