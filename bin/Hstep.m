
function Epsi = Hstep(Theta, fdims, useOneLambdaForAll)


[D, L]       = size(Theta.W);
Epsi         = [];
Epsi.betas   = zeros(1,     L); % ARD prior parameters
Epsi.lambdas = zeros(1, 1 + L); % smoothness prior parameters

%% compute the ARD prior parameters
for l = 1 : L
    wl = Theta.W(:,l);
    Epsi.betas(l) = D / (eps+wl'*wl);
end

%% compute the smoothness prior parameters
if useOneLambdaForAll
    % one lambda for all
    laplacianSquare = [];
    for l = 1 : 1 + L
        if l == 1
            wl = Theta.w0;
        else
            wl = Theta.W(:,l-1);
        end
        
        wl          = reshape(wl, fdims);
        wl_laplace_ = (2*length(fdims)).*del2(wl);
        wl_laplace  = zeros(fdims);
        
        % remove boundary artifacts from del2
        if length(fdims) == 2
            wl_laplace(2:end-1,2:end-1) = wl_laplace_(2:end-1,2:end-1);
        else
            wl_laplace(2:end-1,2:end-1,2:end-1) = wl_laplace_(2:end-1,2:end-1,2:end-1);
        end
        clear wl_laplace_
        
        laplacianSquare(l) = (sum(wl_laplace(:).^2));
    end
    lambda       = D * (L+1) / (sum(laplacianSquare) + eps);
    Epsi.lambdas = lambda .* ones(1, 1 + L);
else
    for l = 1 : 1 + L
        if l == 1
            wl = Theta.w0;
        else
            wl = Theta.W(:,l-1);
        end
        
        wl         = reshape(wl, fdims);
        wl_laplace_ = (2*length(fdims)).*del2(wl);
        wl_laplace  = zeros(fdims);
        
        % remove boundary artifacts from del2
        if length(fdims) == 2
            wl_laplace(2:end-1,2:end-1) = wl_laplace_(2:end-1,2:end-1);
        else
            wl_laplace(2:end-1,2:end-1,2:end-1) = wl_laplace_(2:end-1,2:end-1,2:end-1);
        end
        clear wl_laplace_
        
        laplacianSquare = sum(wl_laplace(:).^2);
        
        Epsi.lambdas(l) = D / (eps+laplacianSquare);
    end
end


