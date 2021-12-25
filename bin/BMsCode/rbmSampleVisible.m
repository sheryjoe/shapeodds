function [pv,v] = rbmSampleVisible(h, W, bv, grid, pretrain)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if nargin < 4, grid     = []; end
if nargin < 5, pretrain = 0;  end
if size(h,1) ~= size(bv,1), bv = repmat(bv, [size(h,1), 1]); end

if isempty(grid)
    pv = computeProbs(h, W, bv, pretrain);
else
    pv = computeProbsShared(h,W, bv, grid, pretrain);
end
if nargout > 1
    if ismatrix(pv)
        v = pv > rand(size(pv));
    else
        v = drawCategorical(pv);
%         v = drawMaxLabel(pv);
    end
end


function pv = computeProbsShared(h,W,bv,grid,pretrain)
pv = zeros(size(bv));
[nHidden,nVisible,nLabels] = size(W);
for j=1:grid.nPatches
    indHidden	= (1:grid.nHiddenPerPatch) + (j-1)*grid.nHiddenPerPatch;
    indVisible	= grid.patches(:,:,j); indVisible = indVisible(:);
    added       = reshape(h(:,indHidden) * reshape(W,nHidden,nVisible*nLabels),size(h,1),nVisible,nLabels);
    pv(:,indVisible,:) = pv(:,indVisible,:) + added;  % visible nodes overlap
end

c = 1; if pretrain == 2, c = 2*c; end;
if nLabels > 1
    pv = softmax(c*pv + bv);
else
    pv = sigmoid(c*pv + bv);
end

function pv = computeProbs(h, W, bv, pretrain)
[nHidden,nVisible,nLabels] = size(W);
pv = reshape(h * reshape(W,nHidden,nVisible*nLabels),size(h,1),nVisible,nLabels);
c = 1; if pretrain == 2, c = 2; end;
if nLabels > 1
    pv = softmax(c*pv + bv);
else
    pv = sigmoid(c*pv + bv);
end

function v = drawCategorical(pv)
[nExamples, nVisible, nLabels] = size(pv);
pcumulative = cumsum(pv, 3);
phit = pcumulative > repmat(rand(nExamples, nVisible), [1 1 nLabels]);
drawCategories = (nLabels+1) - sum(phit,3);

% sometimes you end up with dimensions which are impossible, i.e. every
% category is impossible. in these cases, just choose on category at
% random.
impossibles = find(drawCategories==nLabels+1);
drawCategories(impossibles) = floor(rand(1,numel(impossibles))*nLabels)+1;

% create the output matrix
[x,y] = meshgrid(1:nVisible,1:nExamples);
v     = false(nExamples,nVisible,nLabels); 
v(sub2ind([nExamples,nVisible,nLabels],y(:),x(:),drawCategories(:))) = true; 

function v = drawMaxLabel(pv)
v = false(size(pv));
[~,indMax] = max(pv,[],3);
for i=1:size(pv,3)
    v(1,:,i) = indMax == i;
end














% 
% function [pv,v] = rbmSampleVisible(h,hPos, W, bv, grid, pretrain)
% %UNTITLED Summary of this function goes here
% %   Detailed explanation goes here
% 
% if nargin < 5, grid     = []; end
% if nargin < 6, pretrain = 0;  end
% if size(h,1) ~= size(bv,1), bv = repmat(bv, [size(h,2), 1]); end
% 
% if isempty(grid)
%     pv = computeProbs(h, W, bv, pretrain);
% else
%     pv = computeProbsShared(h, hPos,W, bv, grid, pretrain);
% end
% if nargout > 1
%     if size(pv,3) == 1
%         v = pv > rand(size(pv));
%     else
%         v = drawCategorical(pv);
% %         v = drawMaxLabel(pv);
%     end
% end
% 
% 
% function pv = computeProbsShared(h,hPos,W,bv,grid, pretrain)
% nLabels = size(bv,3);
% pv      = zeros(size(bv));
% [nHidden,nVisible,nLabels] = size(W);
% for j=1:grid.nPatches
% 	indHidden	= (1:grid.nHiddenPerPatch) + (j-1)*grid.nHiddenPerPatch;
% 	h_Hidden	= h(:,indHidden);
% 			
% 	if grid.use_epitome
% 		indVisible	= vec(grid.patches_epi(:,:,j));
% 		for k=1:grid.nTransl
% 			wt				= find(hPos(:,k));
% 			idxsEpitome		= grid.epitome_idxs(:,k);
% 			nInEpitome		= length(idxsEpitome);
% 			try
% 			added(wt,:,:)	= reshape(h_Hidden(wt,:)*reshape(W(:,idxsEpitome,:),...
% 				nHidden,nInEpitome*nLabels),[],nInEpitome,nLabels);
% 			catch
% 				wt;
% 			end
% 		end
% 	else
%     indVisible	= vec(grid.patches(:,:,j));
% 		added				= reshape(h_Hidden* reshape(W,nHidden,nVisible*nLabels),size(h,1),nVisible,nLabels);
% 	end
% 	pv(:,indVisible,:) = pv(:,indVisible,:) + added;
% end
% 
% c = 1; if pretrain == 2, c = 2*c; end;
% if nLabels > 1
%     pv = softmax(c*pv + bv);
% else
%     pv = sigmoid(c*pv + bv);
% end
% 
% function pv = computeProbs(h, W, bv, pretrain)
% % pv = mult3D(h,permute(W,[2,1,3]));
% [nHidden,nVisible,nLabels] = size(W);
% pv = reshape(h * reshape(W,nHidden,nVisible*nLabels),size(h,1),nVisible,nLabels);
% c = 1; if pretrain == 2, c = 2; end;
% if nLabels > 1
%     pv = softmax(c*pv + bv);
% else
%     pv = sigmoid(c*pv + bv);
% end
% 
% 
% 
% function v = drawMaxLabel(pv)
% v = false(size(pv));
% [~,indMax] = max(pv,[],3);
% for i=1:size(pv,3)
%     v(1,:,i) = indMax == i;
% end
% 

