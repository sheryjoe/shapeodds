
function [f_both, f_foreground, f_background, noise_mask_both, noise_mask_foreground, noise_mask_background] = ...
    AddCorrelatedNoise(f, corr_noise_sigma, corrupted_percentage)

if corrupted_percentage == 0
    f_both = f;
    f_foreground = f;
    f_background = f;
    noise_mask_both = zeros(size(f));
    noise_mask_foreground = zeros(size(f));
    noise_mask_background = zeros(size(f));
    return;
end

[X,Y]         = size(f);
XYrand        = randn(X,Y);
window_size   = ceil(5*corr_noise_sigma);
norm_gauss    = normgauss (window_size, window_size, window_size, corr_noise_sigma, 2); % (sz,sz,sgm,dim)
XY_Gconv      = convn(XYrand,norm_gauss,'same');

% mapping to 0-1 range
XY_Gconv      = XY_Gconv - min(XY_Gconv(:));
XY_Gconv      = XY_Gconv ./ max(XY_Gconv(:));

% both
[h,x] = hist(XY_Gconv(:), 100);
cumh = cumsum(h)/sum(h);

indices         = find(cumh >= corrupted_percentage);
both_thresh     = x(indices(1));
noise_mask_both = XY_Gconv<=both_thresh ;
f_both          = xor(f, noise_mask_both);

% foreground
XY_Gconv_foreground   = XY_Gconv.* f;
[h,x] = hist(XY_Gconv_foreground(f(:)==1), 100);
cumh = cumsum(h)/sum(h);

indices              = find(cumh >= corrupted_percentage);
foreground_thresh    = x(indices(1));
noise_mask_foreground = f .* (XY_Gconv_foreground<=foreground_thresh) ;
f_foreground         = xor(f, noise_mask_foreground);

% background
XY_Gconv_background   = XY_Gconv.* (1-f);
[h,x] = hist(XY_Gconv_background(f(:)==0), 100);
cumh = cumsum(h)/sum(h);

indices              = find(cumh >= corrupted_percentage);
background_thresh    = x(indices(1));
noise_mask_background = (1-f) .* (XY_Gconv_background<=background_thresh) ;
f_background         = xor(f, noise_mask_background);

