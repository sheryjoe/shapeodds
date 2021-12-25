function [pv,v, pvall, vall] = gibbsSampling(v,h2,W1,W2,bv,bh1,bh2, opts, nRounds, drawStep, warmup, display)
% v has generally dimensions nExamples x nVisible x nLabels

if isempty(h2) && ~isempty(W2), h2 = rand(1,opts.nHidden2) > 0.5; end
if nargin < 9 , nRounds = 500; end
if nargin < 10, drawStep = 50; end
if nargin < 11, warmup  = 0; end
if nargin < 12, display = false; end

nSamples = floor(nRounds/drawStep);
if nargout > 2 
    pvall = zeros([size(v), nSamples]);
end
if nargin > 3
    vall  = false([size(v), nSamples]);
end

cmap = jet(opts.nLabels);
for i=1:warmup
    [ph1,h1] = rbmSampleHidden1(permute(v, [2 3 1]), W1, bh1,h2,W2,opts.grid);
    [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
end

% if display > 1, then only visualize the part with index display-1
% TODO: correct this
if display
    figure;
    if display > 1
        mask = false(1, prod(opts.imageSize),opts.nLabels);
        mask(1,:,display-1) = true;
        v0 = visible2labels(v .* mask, cmap,opts);
    else
        v0 = visible2labels(v, cmap,opts);
    end
end

count = 0; ticStart = tic;
for i=1:nRounds
    [ph1,h1] = rbmSampleHidden1(permute(v,[2 3 1]), W1, bh1,h2,W2,opts.grid);
    if ~isempty(W2) && ~isempty(bh2)
        [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
        [ph1,h1] = rbmSampleHidden1(permute(v,[2 3 1]),W1,bh1,h2,W2,opts.grid);
    end
    [pv,v] = rbmSampleVisible(h1, W1, bv,  opts.grid);
    if ~mod(i,drawStep)
        if nargout > 2
            count = count + 1;
            pvall(:,:,:,count) = pv; 
            if nargout > 3
                vall(:,:,:,count)  = logical(v); 
            end
        end
        if display
            if display > 1
                pvout = visible2labels(pv .* mask, cmap, opts);
                vout  = visible2labels(v  .* mask, cmap, opts);
            else
                pvout = visible2labels(pv,cmap, opts);
                vout  = visible2labels(v, cmap, opts);
            end            
            subplot(121); imshow(v0,cmap); title('Input');
            subplot(122); imshow(pvout, cmap);
            title(['Reconstruction after ' num2str(i) ' sampling rounds'])
%             subplot(221); imshow(v0, cmap);
%             title('Initial input')
% 			try
% 				subplot(222); imshow(opts.seg,cmap); title('Groundtruth');
% 				subplot(223); imshow(pvout,cmap);
% 				title(['Reconstruction after ' num2str(i) ' sampling rounds'])
%             end            
%             subplot(224); imshow(vout,cmap);
%             title(['Reconstruction labels after ' num2str(i) ' sampling rounds'])
            drawnow;
        end
    end
    progress('Running Gibbs-sampling...',i,nRounds,ticStart,10);
end

% Average probabilities and display ---------------------------------------
% if nargout > 2 || display
%     pvall = pvall / nSums; vall = vall/nSums;
% end
% if display == 1
%     subplot(223); imshow(visible2labels(pvavg,cmap,opts),cmap);
%     title(['Average probability after ' num2str(i) ' sampling rounds'])
% elseif display > 1
%     subplot(223); imshow(visible2labels(pvavg .* mask,cmap,opts),cmap);
%     title(['Average probability after ' num2str(i) ' sampling rounds'])
% end


function labels = visible2labels(v,cmap,opts)
[nExamples, nVisible, nLabels] = size(v);
if nargin < 2
    [~,labels] = max(squeeze(v), [], 3);
elseif nExamples == 1
    labels = reshape(reshape(v,nExamples*nVisible, nLabels) * cmap, [opts.imageSize, 3]);
end












% function [pv,v, pvavg, vavg, ph1rec,ph2rec] = gibbsSampling(v,h2,W1,W2,bv,bh1,bh2, opts, nRounds, warmup, display,diverse,video)
% 
% if isempty(h2) && ~isempty(W2), h2 = rand(1,opts.nHidden2) > 0.5; end
% if nargin < 9 , nRounds = 500; end
% if nargin < 10, warmup  = 0; end
% if nargin < 11, display = false; end
% if nargin < 12, diverse = 0; end
% if nargin < 13, video = 0; end
% 
% cmap = jet(opts.nLabels);
% % v =  reshape(v, [1, prod(opts.imageSize), opts.nLabels]);
% v =  reshape(v, prod(opts.imageSize), opts.nLabels,[]);
% v0 = visible2labels(v, cmap);
% for i=1:warmup
%     [ph1,h1] = rbmSampleHidden1(v, W1, bh1,h2,W2,opts.grid);
%     [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
% end
% 
% % if display > 1, then only visualize the part with index display-1
% drawStep = 50; nSums = 0;
% if display
%     figure;
%     if video
%         set(gcf,'Color','white'); axis tight; 
%         set(gca, 'nextplot','replacechildren', 'Visible','off');
%         vidObj = VideoWriter('gibbsSamplingCow.avi');
%         vidObj.Quality = 100;
%         vidObj.FrameRate = 10;
%         open(vidObj);
%     end
%     if display > 1
%         mask = false(1, prod(opts.imageSize),opts.nLabels);
%         mask(1,:,display-1) = true;
%         v0 = visible2labels(v .* mask, cmap);
%     end
% end
% 
% if nargin > 2 || display
%     pvavg = zeros(size(v));
%     vavg  = zeros(size(v));
% end
% if nargin > 4
%     ph1rec = zeros(floor(nRounds/drawStep), opts.nHidden1);
%     if ~isempty(W2) && ~isempty(bh2)
%         ph2rec = zeros(floor(nRounds/drawStep), opts.nHidden2);
%     end
% end
% h1_prev= []; pretrain = 0;
% for i=1:nRounds
%     [ph1,h1,hPos] = rbmSampleHidden1(v, W1, bh1,h2,W2,opts.grid,pretrain,diverse,h1_prev);
%     if ~isempty(W2) && ~isempty(bh2)
%         [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
%         [ph1,h1,hPos] = rbmSampleHidden1(v,W1,bh1,h2,W2,opts.grid,pretrain,diverse,h1_prev);
%     end
%     [pv,v] = rbmSampleVisible(h1,hPos, W1, bv,  opts.grid);
%     v = permute(v,[2,3,1]); pv = permute(pv,[2,3,1]);
%     if (~mod(i,drawStep))&&(i>drawStep*10)
%         h1_prev = [h1_prev;h1];
%         if nargin > 2 || display
%             pvavg = pvavg + pv; nSums = nSums + 1;
%             vavg  = vavg + double(v); 
%         end
%         if nargin > 4
%             ph1rec(nSums,:) = ph1;
%             if ~isempty(W2) && ~isempty(bh2), ph2rec(nSums,:) = ph2; end
%         end
%         if display
%             if display > 1
%                 pvout = visible2labels(pv .* mask, cmap);
%                 vout  = visible2labels(v  .* mask, cmap);
%             else
%                 pvout = visible2labels(pv,cmap);
%                 vout  = visible2labels(v, cmap);
%             end            
%             subplot(221); imshow(reshape(v0, [opts.imageSize, 3]), cmap);
%             title('Initial input')
% 			try
% 				subplot(222); imshow(opts.seg,cmap); title('Groundtruth');
% 				subplot(223); imshow(reshape(pvout,[opts.imageSize, 3]),cmap);
% 				title(['Reconstruction after ' num2str(i) ' sampling rounds'])
%             end            
%             subplot(224); imshow(reshape(vout,[opts.imageSize, 3]),cmap);
%             title(['Reconstruction labels after ' num2str(i) ' sampling rounds'])
%             drawnow;
%             if video, writeVideo(vidObj, getframe(gcf)); end
%         end
%     end
% end
% 
% % Average probabilities and display ---------------------------------------
% if nargin > 2 || display
%     pvavg = pvavg / nSums; vavg = vavg/nSums;
% end
% if display == 1
%     subplot(223); imshow(reshape(visible2labels(pvavg,cmap),[opts.imageSize, 3]),cmap);
%     title(['Average probability after ' num2str(i) ' sampling rounds'])
%     if video, writeVideo(vidObj, getframe(gcf)); close(vidObj);end
% elseif display > 1
%     subplot(223); imshow(reshape(visible2labels(pvavg .* mask,cmap),[opts.imageSize, 3]),cmap);
%     title(['Average probability after ' num2str(i) ' sampling rounds'])
% end
% 
% 
% function labels = visible2labels(v,cmap)
% if nargin < 2
%     [~,labels] = max(squeeze(v), [], 3);
% else
%     labels = squeeze(v) * cmap;
% end
% 
