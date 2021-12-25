
function VisImageSetWithOverlay2(fs, qs, rows, cols, figure_title, vectorized, hfig)

if ~exist('vectorized' , 'var')
    vectorized = 1;
end
if ~exist('hfig' , 'var')
    hfig = [];
end

if vectorized
    nSamples = size(fs, 2);
else
    nSamples = size(fs, 3);
end

fs_imgs = [];
qs_imgs = [];
for ii = 1 : nSamples
    if vectorized
        %img = reshape_rowwise(fs(:,ii), rows, cols);
        img = reshape(fs(:,ii), rows, cols);
        %q   = reshape_rowwise(qs(:,ii), rows, cols);
        q   = reshape(qs(:,ii), rows, cols);
    else
        img = fs(:,:,ii);
        q   = qs(:,:,ii);
    end
    
    fs_imgs(:,:,1,ii) = img;
    fs_imgs(:,:,2,ii) = q;
    %fs_imgs(:,:,3,ii) = img; %zeros(size(qmap));
    fs_imgs(:,:,3,ii) = zeros(size(q));
end

if isempty(hfig)
    figure;
end
montage(fs_imgs, 'DisplayRange',[]);
title(figure_title);
