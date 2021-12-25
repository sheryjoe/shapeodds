function Wvec = UpdateModelParams(Wvec, timeStep, descentDirection, updateParams)

if ~updateParams.useSmoothness
    Wvec = Wvec + timeStep .* descentDirection;
    return
end

% tic
ispar    = updateParams.ispar;
nworkers = updateParams.nworkers;

% latent space dimensionality
L     = updateParams.L;

if L == 0
    ispar = 0;
end 

% observed space dimensionality
D     = updateParams.D;

% sample dimensions
fdims = updateParams.fdims;

% prior type
priorType = updateParams.priorType;

% prior hyperparameters
lambdas     = updateParams.Epsi.lambdas; % 1 x 1 + L

% the current bias and loading matrix
Wtilde = reshape(Wvec, [D,1 + L]); % the first column is the offset vector

% convert them into maps
Wtilde = UnVectorizeMaps(Wtilde, fdims, 1 + L);

% the gradient of the log-likelihood lower bound
% descentDirection = -gradient = --dL_dwl = dL_dwl
descentDirection = reshape(descentDirection, [D,1 + L]);

% convert them into maps
descentDirection = UnVectorizeMaps(descentDirection, fdims, 1 + L);

% pad zeros in case of unsquare matrices
isSquared = 0;
if mean(fdims) ~= fdims(1)
    % images are not square, make them square for isotropic filter design and need
    % to report model parameters wrt the original size
    D_orig     = D;
    fdims_orig = fdims;
    
    [Wtilde, D, fdims]       = SquareMaps(Wtilde, fdims_orig, 1 + L);
    [descentDirection, ~, ~] = SquareMaps(descentDirection, fdims_orig, 1 + L);
  
    isSquared = 1;
end

if ispar
    
    poolobj = gcp('nocreate');
    if isempty(poolobj)
        myCluster            = parcluster('local');
        myCluster.NumWorkers = nworkers;
        parpool('local', nworkers);
    end
    
    wl_store = cell(1,1+L);
    for l = 1 : 1 + L
        wl_store{l} = GetNthMap(Wtilde,l);
    end
    
    %parfor_progress(1+L); % Initialize
    
    % apply the impicit scheme update
    parfor l = 1 : 1 + L
        
        %parfor_progress; % Count
        
        % current vector
        wl     = wl_store{l};
        %wl = GetNthMap(Wtilde,l);
        
        % its descent direction
        dL_dwl = GetNthMap(descentDirection, l);
        
        % prior filter design
        A  = ComputeImplicitSchemeFilter(priorType, fdims, lambdas(l),  timeStep);
        
        %apply the update in the fourier domain the go back
        %wl  = real(ifftn(fftn(wl + timeStep .* dL_dwl) .* A,  'symmetric'));
        wl  = real(ifftn(fftn(wl + timeStep .* dL_dwl) .* A,  'nonsymmetric'));
        
        % update the loading matrix
        wl_store{l} = wl;
        %Wtilde = SetNthMap(Wtilde, wl, l);
    end
    
    %parfor_progress(0); % Clean up
    
    % update the loading matrix
    for l = 1 : 1 + L
        Wtilde = SetNthMap(Wtilde, wl_store{l}, l);
    end
else
    
    % apply the impicit scheme update
    if L == 0
        % current vector
        wl = Wtilde;
        
        % its descent direction
        dL_dwl = descentDirection;
        
        % prior filter design
        A  = ComputeImplicitSchemeFilter(priorType, fdims, lambdas(1),  timeStep);
        %A  = ComputeImplicitSchemeFilter(priorType, min(fdims)*ones(1, length(fdims)), lambdas(l),  timeStep);
        
        %apply the update in the fourier domain then go back
        %wl  = real(ifftn(fftn(wl + timeStep .* dL_dwl) .* A,  'symmetric'));
        wl  = real(ifftn(fftn(wl + timeStep .* dL_dwl) .* A,  'nonsymmetric'));
        
        % update the loading matrix
        Wtilde = wl;
        
    else
        for l = 1 : 1 + L
            % current vector
            wl = GetNthMap(Wtilde,l);
            
            % its descent direction
            dL_dwl = GetNthMap(descentDirection, l);
            
            % prior filter design
            A  = ComputeImplicitSchemeFilter(priorType, fdims, lambdas(l),  timeStep);
            %A  = ComputeImplicitSchemeFilter(priorType, min(fdims)*ones(1, length(fdims)), lambdas(l),  timeStep);
            
            %apply the update in the fourier domain then go back
            %wl  = real(ifftn(fftn(wl + timeStep .* dL_dwl) .* A,  'symmetric'));
            wl  = real(ifftn(fftn(wl + timeStep .* dL_dwl) .* A,  'nonsymmetric'));
            
            % update the loading matrix
            Wtilde = SetNthMap(Wtilde, wl, l);
        end
    end
    
end

% vectorize the maps
Wtilde = VectorizeMaps(Wtilde, D, 1 + L);

% take care of the image padding for non-square label maps
if isSquared
    Wtilde = UnsquareModelParameters2(Wtilde, fdims, fdims_orig, D_orig);
end

% returning to the vector form to optimization process
Wvec = Wtilde(:);


% elapsedTime = toc;
% fprintf('Mstep: UpdateModelParams took %f seconds ...\n', elapsedTime);