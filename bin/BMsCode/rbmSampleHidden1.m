function [ph,h] = rbmSampleHidden1(v, W1, bh, h2, W2, grid, pretrain)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% 
% v must be in format of 
% [nVisible,nLabels,nExamples]
if nargin < 4 || isempty(h2), h2 = 0;  end
if nargin < 5 || isempty(W2), W2 = 0;  end
if nargin < 6, grid     = [];          end
if nargin < 7, pretrain = 0;           end

if size(v,3) ~= size(bh,1), bh = repmat(bh, [size(v,3), 1]); end

% Input from hidden layer 2
if (isscalar(h2) && W2 == 0) || (isscalar(h2) && h2 == 0)
    h2in = 0;
else
    if size(v,3) ~= size(h2,1), h2 = repmat(h2, [size(v,3), 1]); end
    h2in = h2 * W2';
end

% input from visible units
vin  = inputFromVisible(v, W1, grid); 

% Combine all inputs 
c = 1; if pretrain == 1, c = 2; end
ph   = sigmoid(c*vin + h2in + bh); % What if everything is 0?
if nargout > 1
    h = ph > rand(size(ph));    % should I do this by repmatting a single random sample?
end
    
function vin = inputFromVisible(v, W1, grid)  
if isempty(v) || isempty(W1)
    vin = 0; 
    return 
end

[nHidden,nVisible,nLabels] = size(W1); nExamples = size(v,3);
if isempty(grid)
    vin = (reshape(W1,nHidden,nVisible*nLabels) * reshape(v,nVisible*nLabels, []))';
else
    vin = zeros(nExamples,nHidden);
    W1r = reshape(W1,nHidden,nVisible*nLabels);
    for j=1:grid.nPatches
        indVisible = grid.patches(:,:,j); indVisible = indVisible(:);
        indHidden = (1:grid.nHiddenPerPatch) + (j-1)*grid.nHiddenPerPatch;
        vin(:,indHidden) = (W1r * reshape(v(indVisible,:,:),nnz(indVisible)*nLabels, []))';
%         for i=1:nLabels
%             vin(:,indHidden) = vin(:,indHidden) + v(:,indVisible,i) * W1(:,:,i);
%         end
    end
end













% function [ph,h,hPos] = rbmSampleHidden1(v, W1, bh, h2, W2, grid, pretrain,diverse,h1_prev)
% %UNTITLED Summary of this function goes here
% %   Detailed explanation goes here
% 
% if nargin < 4 || isempty(h2), h2 = 0;  end
% if nargin < 5 || isempty(W2), W2 = 0;  end
% if nargin < 6, grid     = [];          end
% if nargin < 7, pretrain = 0;           end
% if nargin < 8, diverse  = 0;           end
% 
% hPos = [];
% % v must be in format of 
% % [nvisible,nlabels,ntrain]
% % assert sizes are compatibe
% [nvisible_v,nlabels_v,ntrain_v ] = size(v);
% [nhidden_w,nvisible_w,nlabels_w] = size(W1);
% %assert(nvisible_w == nvisible_v);
% assert(nlabels_w  == nlabels_v);
% 
% if size(v,3) ~= size(bh,1), bh = repmat(bh, [size(v,3), 1]); end
% 
% c = 1; if pretrain == 1, c = 2; end
% [input_for_hidden]  = inputFromVisible(v, W1, bh, grid);
% h2in				= h2 * W2';
% if grid.use_epitome
% 	[input_for_hidden,hPos] = rbmChoosePosition(input_for_hidden,h2in,bh,grid,c);
% end
% input_total = c*input_for_hidden + h2in + bh;
% %
% ph					= sigmoid(input_total);
% 
% if nargout > 1
%     % Iasonas' code diversity ---------------------------------------------
%     if diverse && ~isempty(h1_prev)
%         nsamp = 200;
%         h			= bsxfun(@lt,rand(nsamp,length(ph)),ph);	
%         %prb   = 1;
%         dst_tot= 0 ;
%         for d = [1:size(h1_prev,1)]
%             if d==2
%                 d;
%             end
%             dfs 	= bsxfun(@minus,h,h1_prev(d,:));
%             %dst		= sum(dfs.*dfs,2);
%             dst		= sum(abs(dfs),2);
%             dst_tot		= dst_tot+ diverse*dst;
%         end
%         maxD = max(dst_tot); dst_tot = dst_tot - maxD;
%         prb = exp(dst_tot)./sum(exp(dst_tot));
%         samp	= randsample(nsamp,1,true,prb);
%         h			= h(samp,:);
%     else
%         h = ph > rand(size(ph));
%     end
% end
%     
% 
