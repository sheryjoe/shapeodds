
function VisImageSet2(fs, rows, cols, figure_title, vectorized, hfig)

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
for ii = 1 : nSamples
    if vectorized
        img = reshape(fs(:,ii), rows, cols);
    else
        img = fs(:,:,ii);
    end
    
    fs_imgs(:,:,1,ii) = img;
end

if isempty(hfig)
    figure;
end
if nSamples == 1
    imshow(fs_imgs, []);
else
    montage(fs_imgs, 'DisplayRange',[]);
end
title(figure_title);
