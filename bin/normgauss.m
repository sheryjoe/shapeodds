function norm_gauss = normgauss(KS1,KS2,KS3,sigma,COG)
% KS = 5; % window size = 2*KS+1
% sigma = 3; 
% COG = 1;  % number of dimensions
% For N-D gaussian case
% the formulation is 1/(sqrt(2pi)sigma)^N.(e^-(xvec^2)/2sigma^2)
if (COG == 1)
    % creating a gaussian 1D function
    X = -KS1:1:KS1;
    exponent =  (1/(2*sigma^2)) * (X.^2);
    val = exp(-exponent);
    norm_gauss = val/sum(val(:));
elseif (COG == 2)    
    % creating a gaussian 2D function
    [X Y] = meshgrid(-KS1:1:KS1,-KS2:1:KS2);
    exponent  = (1/(2*sigma^2)) * (X.^2 + Y.^2);
    val = exp(-exponent);
    norm_gauss = val/sum(val(:));
 else
    % creating a gaussian 3D function
    [X Y Z] = meshgrid(-KS1:1:KS1,-KS2:1:KS2,-KS3:1:KS3);
    exponent  = (1/(2*sigma^2)) * (X.^2 + Y.^2+Z.^2);
    val = exp(-exponent);
    norm_gauss = val/sum(val(:));
end    

% % creating a gaussian 3D function
% KS0 = 2;
% KS1 = 2;
% KS2 = 5;
% sigma0 = 1;
% sigma1 = 1;
% sigma2 = 2;
% [X Y Z] = meshgrid(-KS0:1:KS0,-KS1:1:KS1,-KS2:1:KS2);
% exponent  = (1/(2*sigma0^2)) * (X.^2) + ...
%             (1/(2*sigma1^2)) * (Y.^2) + ...
%             (1/(2*sigma2^2)) * (Z.^2);
% val = exp(-exponent);
% norm_gauss = val/sum(val(:));

