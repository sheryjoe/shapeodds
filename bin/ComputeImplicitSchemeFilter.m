
function [A] = ComputeImplicitSchemeFilter(priorType, fdim, prior_lambda, timeStep, useCosines)

if ~exist('useCosines', 'var') % this gives a weird stair case effect (might need larger stencil?)
    useCosines = 0;
end

switch length(fdim)
    case 2, % 2D
        %N_x = fdim(1); N_y = fdim(2);
        N_x = fdim(2); N_y = fdim(1);
        switch priorType,
            case 'dirichlet',
                
                % frequency variables
                u   = (0:N_x-1)';
                v   = (0:N_y-1)';
                
                [u,v] = meshgrid(u,v);
                
                H_u  = sin((pi/N_x) .* u).^2 ;
                H_v  = sin((pi/N_y) .* v).^2 ;
                H_uv = H_u + H_v;
                
                A  = 1 ./ (1 + 4 * prior_lambda .* timeStep .* H_uv);
                
                clear u v H_uv H_u H_v
                
            case 'laplacian-square',
                
                % frequency variables
                u   = (0:N_x-1)';
                v   = (0:N_y-1)';
                
                [u,v] = meshgrid(u,v);
                
                if useCosines
                    HB = 20 - 16*cos(u) - 16*cos(v) + 2*cos(2*u) + 2*cos(2*v) + 4*cos(u+v) + 4*cos(u-v);
                else
                    % biharmonic/bilaplacian filter
                    jpiMu = 1i*(pi/N_x).*(u - N_x/2);
                    jpiNv = 1i*(pi/N_y).*(v - N_y/2);
                    
                    HB = 20 ...
                        - 8.*exp(-jpiMu) - 8.*exp(jpiMu) - 8.*exp(-jpiNv) - 8.*exp(jpiNv) ...
                        + exp(-2*jpiMu) + exp(2*jpiMu)   + exp(-2*jpiNv) + exp(2*jpiNv)  ...
                        + 2.*exp(-jpiMu).*exp(-jpiNv)    + 2.*exp(jpiMu).*exp(jpiNv) ...
                        + 2.*exp(-jpiMu).*exp(jpiNv)     + 2.*exp(jpiMu).*exp(-jpiNv) ;
                end
                
                A     = 1 ./ (1 + prior_lambda .* timeStep .* HB);
                A     = fftshift(A);
                %A     = A ./ max(A(:));
                
                clear u v HB jpiMu jpiNv
        end
        
    case 3,
        N_x = fdim(1); N_y = fdim(2); N_z = fdim(3);
        switch priorType,
            case 'dirichlet',
                
                % frequency variables
                u   = (0:N_x-1)';
                v   = (0:N_y-1)';
                w   = (0:N_z-1)';
                
                [u,v,w] = meshgrid(u,v,w);
                
                H_u   = sin((pi/N_x) .* u).^2 ;
                H_v   = sin((pi/N_y) .* v).^2 ;
                H_w   = sin((pi/N_z) .* w).^2 ;
                H_uvw = H_u + H_v + H_w;
                
                A  = 1 ./ (1 + 4 * prior_lambda .* timeStep .* H_uvw);
                
                clear u v H_uvw H_u H_v H_w
                
            case 'laplacian-square',
                
                % frequency variables
                pixelspacing = [1 1 1];
                [u,v,w] = ifftshiftedcoormatrix3([N_x N_y N_z] );
                u = double(u/N_x/pixelspacing(1));
                v = double(v/N_y/pixelspacing(2));
                w = double(w/N_z/pixelspacing(3));
                
                if useCosines
                    % biharmonic/bilaplacian filter
                    HB = 42 - 24*cos(u) - 24*cos(v) - 24*cos(w) + 2*cos(2*u) + 2*cos(2*v) + 2* cos(2*w) + 4*cos(u+v) + 4*cos(v+w) + 4*cos(u+w)...
                        + 4*cos(u-v) + 4*cos(v-w) + 4*cos(u-w);
                else
                    % biharmonic/bilaplacian filter
                    jpiMu = 1i.*u;
                    jpiNv = 1i.*v;
                    jpiQw = 1i.*w;
                    
                    %based on 6point laplacian stencil (not stable)
                    HB = 42 ...
                        - 12 .* exp(-jpiMu) - 12 .* exp(jpiMu) ...
                        - 12 .* exp(-jpiNv) - 12 .* exp(jpiNv) ...
                        - 12 .* exp(-jpiQw) - 12 .* exp(jpiQw) ...
                        + exp(-2.*jpiMu) + exp(2.*jpiMu) ...
                        + exp(-2.*jpiNv) + exp(2.*jpiNv) ...
                        + exp(-2.*jpiQw) + exp(2.*jpiQw) ...
                        + 2 .* exp(-jpiQw) .* exp(-jpiMu) + 2 .* exp(jpiQw) .* exp(jpiMu) + 2 .* exp(-jpiQw) .* exp(jpiMu) + 2 .* exp(jpiQw) .* exp(-jpiMu) ...
                        + 2 .* exp(-jpiQw) .* exp(-jpiNv) + 2 .* exp(jpiQw) .* exp(jpiNv) + 2 .* exp(-jpiQw) .* exp(jpiNv) + 2 .* exp(jpiQw) .* exp(-jpiNv) ...
                        + 2 .* exp(-jpiMu) .* exp(-jpiNv) + 2 .* exp(jpiMu) .* exp(jpiNv) + 2 .* exp(-jpiMu) .* exp(jpiNv) + 2 .* exp(jpiMu) .* exp(-jpiNv) ;
                end
                
                A  = 1 ./ (1 + prior_lambda .* timeStep .* HB);
                A  = A ./ max(A(:));
                
                clear u v w jpiMu jpiNv jpiQw HB
        end
end

% neuman condition
if max(abs(A(:))) > 1
    close all
    fprintf( 'ERROR: unstable choice of time step (and/or lambda) ... halting ...\n');
    return;
end
