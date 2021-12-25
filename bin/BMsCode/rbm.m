function [info, Wcnn, W1, biasVisible, biasHidden1, W2, biasHidden2] = ...
    rbm(data, opts, Wcnn, W1, biasVisible, biasHidden1, W2, biasHidden2)

[nExamples, nVisibleAll, nLabels] = size(data);
assert(nLabels == opts.nLabels, 'Number of labels does not match opts.nLabels')

% Initializations ---------------------------------------------------------
rng(0);
if isempty(opts.grid)
    nVisible = nVisibleAll;
    nHidden1 = opts.nHidden1;
    if nargin < 4, W1 = opts.W_0*randn(nHidden1, nVisible, nLabels); end
    if nargin < 5 || isempty(biasVisible), biasVisible = zeros(1, nVisible, nLabels); end
    if nargin < 6 || isempty(biasHidden1), biasHidden1  = zeros(1, nHidden1);  end
    gradWeights1    = zeros(nHidden1, nVisible, nLabels);
    gradBiasVisible = zeros(1, nVisible, nLabels);
    gradBiasHidden1 = zeros(1, nHidden1);
    if opts.doDeep
        nHidden2 = opts.nHidden2;
        if nargin < 7 || isempty(W2), W2  = opts.W_0*randn(nHidden1, nHidden2); end
        if nargin < 8 || isempty(biasHidden2), biasHidden2 = zeros(1, nHidden2); end
        gradWeights2    = zeros(nHidden1, nHidden2);
        gradBiasHidden2 = zeros(1, nHidden2);
    else
        W2 = []; biasHidden2 = []; 
    end
else  
    nVisible    = opts.grid.nVisiblePerPatch;
    nHidden1    = opts.grid.nHiddenPerPatch;
    nHidden1All = opts.grid.nHiddenPerPatch * opts.grid.nPatches;
    if nargin < 4 || isempty(W1), W1 = opts.W_0*randn(nHidden1,nVisible, nLabels); end
    if nargin < 5 || isempty(biasVisible), biasVisible = zeros(1, nVisibleAll, nLabels); end
    if nargin < 6 || isempty(biasHidden1), biasHidden1 = zeros(1, nHidden1All);  end
    gradWeights1    = zeros(nHidden1,nVisible, nLabels);
    gradBiasVisible = zeros(1, nVisibleAll, nLabels);
    gradBiasHidden1 = zeros(1, nHidden1All);
    if opts.doDeep
        nHidden2 = opts.nHidden2;
        if nargin < 7 || isempty(W2), W2  = opts.W_0*randn(nHidden1All, nHidden2); end
        if nargin < 8 || isempty(biasHidden2), biasHidden2 = zeros(1, nHidden2); end
        gradWeights2    = zeros(nHidden1All, nHidden2);
        gradBiasHidden2 = zeros(1, nHidden2);
    else
        W2 = []; biasHidden2 = []; 
    end
end
if opts.doCNN
    features = opts.cnnScoresTrain;
    mn = min(features,[],3); mx = max(features,[],3);
    features = 2*bsxfun(@rdivide,bsxfun(@minus, features, mn), mx-mn)-1;
    featDim  = size(features,3);
    if nargin < 3 || isempty(Wcnn), 
        Wcnn = opts.W_0*randn(featDim, nLabels); 
%         Wcnn = opts.W_0*(2*eye(featDim,nLabels) - 1);
    end
    gradWeightsCnn = zeros(featDim, nLabels);
else
    Wcnn = []; features = [];
end
if opts.PCD
    hPersistent = rand(opts.batchSize,nHidden1);
end

opts.batchSize = min(opts.batchSize, nExamples);
dataPerm = permute(data, [2 3 1]); % keep permuted version of the data for efficiency
if opts.display, fh = figure; end  % do this now to avoid matlab taking control of your desktop
ticStart  = tic;
for epoch = 1:opts.nEpochs
    if epoch > opts.epochChangeMomentum, opts.momentum = 0.9; end,
    info.errorTrain(epoch) = 0;
    info.errorVal(epoch)   = 0;
    if opts.batchSize > 0 && opts.batchSize < nExamples
        indPerm  = randperm(nExamples);
        data     = data(indPerm,:,:);
        dataPerm = dataPerm(:,:,indPerm);
        if opts.doCNN, features = features(indPerm,:,:); end
    end
    for ex = 1:opts.batchSize:nExamples
        batch     = data(ex:min(ex+opts.batchSize-1, nExamples), :, :);
        batchPerm = dataPerm(:, :,ex:min(ex+opts.batchSize-1, nExamples));
        batchSize = size(batch,1); % batchSize != opts.batchSize
        if opts.doCNN
            cnnScores = features(ex:min(ex+opts.batchSize-1, nExamples), :,:);
        end
        if ~mod(ceil(ex/opts.batchSize), 10)
            fprintf('Epoch %d/%d, batch %d/%d, using %d examples\n', epoch, opts.nEpochs,...
                ceil(ex/opts.batchSize), ceil(nExamples/opts.batchSize), batchSize);
        end
        vWake     = batch;      % batchSize x nVisible x nLabels
        vWakePerm = batchPerm;  % nVisible x nLabels x batchSize
        bh = repmat(biasHidden1, [batchSize, 1]); % batchSize x nHidden
        bv = repmat(biasVisible, [batchSize, 1]); % batchSize x nVisible x nHidden
        
        % Wake phase ------------------------------------------------------
        if opts.doDeep
            bh2	= repmat(biasHidden2, [batchSize, 1]); % batchSize x nHidden2
            [phWake, ph2Wake] = meanField(vWakePerm,W1,W2,bh,bh2,opts);
            hWake = prob2state(phWake);
            if isempty(opts.grid)
                wake = mult3D(phWake', vWake);
            else
                wake = expectationShared(vWakePerm,phWake,opts); % nHidden x nVisible x nLabels    
            end
            wake2 = hWake' * ph2Wake;   % nHidden x nHidden2
        else
            % P(h_i|v). We double visible states if training the bottom layer of a DBM
            % (See Deep Boltzmann Machines - Salakhutdinov. Hinton - 2009)
            [phWake,hWake] = rbmSampleHidden1(vWakePerm,W1,bh,0,0,opts.grid,opts.pretrain);
            if isempty(opts.grid)
                wake = mult3D(phWake', vWake);  % nHidden x nVisible x nLabels
            else
                wake = expectationShared(vWakePerm,phWake,opts);
            end
        end
        
        % Dream phase -----------------------------------------------------
        % 'dream' phase (intractable - solve with Contrastive Divergence)
        % <h_i v_j>_{P(h,v)} ~= sum_{h = 1} h * P(h_i = label_h|v^K) * v^K_j
        if opts.doCNN
            bv = bv + reshape(reshape(cnnScores, batchSize*nVisible,featDim)*Wcnn,...
                batchSize,nVisible,nLabels);
        end
        if opts.PCD 
            hWake = hPersistent; 
        end
        if opts.doDeep
            [~, vDream,phDream,hDream,ph2Dream] = ...
                contrastiveDivergenceDeep(hWake,W1,W2,bv,bh,bh2,opts);
            dream2 = hDream' * ph2Dream;
        else
            [~,vDream,phDream,hDream] = contrastiveDivergence(hWake,W1,bv,bh,opts);
        end
        if opts.PCD
            hPersistent = hDream;
        end
        if isempty(opts.grid)
            dream = mult3D(phDream', vDream); % nHidden x nVisible x nLabels
        else
            dream = expectationShared(permute(vDream,[2 3 1]),phDream,opts);
        end
        
        % Update weights and biases ---------------------------------------
        gradWeights1     = opts.momentum * gradWeights1...
                         + (opts.epsw(epoch)/batchSize) * (wake - dream) ...
                         - opts.epsw(epoch) * opts.weightCost * W1;
        gradBiasVisible  = opts.momentum * gradBiasVisible ...
                         + (opts.epsbv(epoch)/batchSize) * sum(vWake - vDream);
        gradBiasHidden1  = opts.momentum * gradBiasHidden1...
                         + (opts.epsbh(epoch)/batchSize) * sum(phWake - phDream);
        W1          = W1          + gradWeights1;
        biasHidden1 = biasHidden1 + gradBiasHidden1;
        biasVisible = biasVisible + gradBiasVisible;
        if opts.doDeep
            gradWeights2    = opts.momentum * gradWeights2...
                            + (opts.epsw(epoch)/batchSize) * (wake2 - dream2) ...
                            - opts.epsw(epoch)*opts.weightCost * W2;
            gradBiasHidden2 = opts.momentum * gradBiasHidden2...
                            + (opts.epsbh(epoch)/batchSize) * sum(ph2Wake - ph2Dream);
            W2              = W2          + gradWeights2;
            biasHidden2     = biasHidden2 + gradBiasHidden2;
        end
        if opts.doCNN
            wakeCnn  = reshape(cnnScores, [],nLabels)' * reshape(vWake, [], nLabels);
            dreamCnn = reshape(cnnScores, [],nLabels)' * reshape(vDream,[], nLabels);
            gradWeightsCnn  = opts.momentum * gradWeightsCnn...
                            + (opts.epswcnn(epoch)/batchSize) * (wakeCnn - dreamCnn) ...
                            - opts.epswcnn(epoch)*opts.weightCost * Wcnn;
            Wcnn            = Wcnn        + gradWeightsCnn;
        end
        
        if opts.PCD
            [~,h] = rbmSampleHidden1(vWakePerm,W1,biasHidden1,[],[],opts.grid,opts.pretrain);
            [~,v] = rbmSampleVisible(h,W1,bv,opts.grid,opts.pretrain);
            info.errorTrain(epoch) = info.errorTrain(epoch) + ...
                sum((vWake(:)-v(:)).^2)/nExamples;            
        else
            info.errorTrain(epoch) = info.errorTrain(epoch) + ...
                sum((vWake(:)-vDream(:)).^2)/nExamples;
        end
        if opts.display && ~mod(epoch, opts.displayHiddenActivations)
            clf(fh); imagesc(phDream); title('Hidden unit activations');drawnow;
        end
    end
    
    [info,opts,pvt,pvv] = evaluateOnValidationSet(opts,info,data,features,epoch,...
        W1,W2,Wcnn,biasVisible,biasHidden1,biasHidden2);
    if opts.display
        displayProgress(info,opts,epoch,fh,pvt,pvv,W1);
    end
    
    if opts.saveStep && (epoch == opts.nEpochs || ~mod(epoch, opts.saveStep))
        save([opts.saveFile],'W1','W2','Wcnn','biasVisible','biasHidden1',...
            'biasHidden2','opts','info');
    end
    
    %     if opts.verbose
    %         progress(sprintf('Training RBM for %s, epoch %d/%d, error: %.5f...',...
    %             opts.objectClass, epoch, opts.nEpochs,info.errorTrain(epoch)),...
    %             epoch, opts.nEpochs, ticStart, 0);
    %     end
    if opts.verbose
        progress(sprintf('Training RBM for %s, epoch %d/%d, error: %.5f...',...
            opts.verbose_text, epoch, opts.nEpochs,info.errorTrain(epoch)),...
            epoch, opts.nEpochs, ticStart, 0);
    end
end


% -------------------------------------------------------------------------
function [info,opts,pvt,pvv] = evaluateOnValidationSet(opts,info,data,features,epoch,W1,W2,Wcnn,bv,bh1,bh2)
% -------------------------------------------------------------------------
pvt = [];pvv = [];
if isfield(opts,'dataVal') && ~isempty(opts.dataVal)
    [nExVal,nVisibleAll,nLabels] = size(opts.dataVal); featDim = size(Wcnn,1);
    if ~isfield(opts,'dataTrain')
        opts.dataTrain = data(1:nExVal,:,:);
        if opts.doCNN
            opts.cnnScoresTrain = features(1:nExVal,:,:);
        end
    end
    if opts.doCNN
        bvt = bsxfun(@plus, bv, reshape(reshape(opts.cnnScoresTrain, ...
            nExVal*nVisibleAll,featDim)*Wcnn, nExVal,nVisibleAll,nLabels));
        bvv = bsxfun(@plus, bv, reshape(reshape(opts.cnnScoresVal, ...
            nExVal*nVisibleAll,featDim)*Wcnn, nExVal,nVisibleAll,nLabels));
    else
        bvt = repmat(bv, [nExVal, 1]); bvv = bvt;
    end
    if opts.doDeep  % Maybe use CD or gibbs Sampling code for evaluation
        h2 = rand(1,opts.nHidden2) > 0.5;
        % Val reconstructions
        [ph1,h1] = rbmSampleHidden1(permute(opts.dataVal,[2 3 1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
        [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
        [ph1,h1] = rbmSampleHidden1(permute(opts.dataVal,[2,3,1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
        [pvv,vv] = rbmSampleVisible(h1,W1,bvv,opts.grid,opts.pretrain);
        % Train reconstructions
        [ph1,h1] = rbmSampleHidden1(permute(opts.dataTrain,[2 3 1]) ,W1,bh1,h2,W2,opts.grid,opts.pretrain);
        [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
        [ph1,h1] = rbmSampleHidden1(permute(opts.dataTrain,[2 3 1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
        [pvt,vt] = rbmSampleVisible(h1,W1,bvt,opts.grid,opts.pretrain);
    else
        % Val reconstructions
        [ph,h]   = rbmSampleHidden1(permute(opts.dataVal,[2 3 1]),W1,bh1,0,0,opts.grid,opts.pretrain);
        [pvv,vv] = rbmSampleVisible(h,W1,bvv,opts.grid,opts.pretrain);
        % Train reconstructions
        [ph,h]   = rbmSampleHidden1(permute(opts.dataTrain,[2 3 1]),W1,bh1,0,0,opts.grid,opts.pretrain);
        [pvt,vt] = rbmSampleVisible(h,W1,bvt,opts.grid,opts.pretrain);
    end
    info.errorVal(epoch) = sum((opts.dataVal(:)-vv(:)).^2) / nExVal;
    info.llTrain(epoch)  = computeLikelihood(pvt,opts.dataTrain,opts);
    info.llVal(epoch)    = computeLikelihood(pvv,opts.dataVal,opts);
%     info.freeEnergyDiff(epoch) = mean(freeEnergy(vt,W1,bv,bh1,opts) - freeEnergy(vv,W1,bv,bh1,opts));
end

% -------------------------------------------------------------------------
function displayProgress(info,opts,epoch,fh,pvTrain,pvVal,W1)
% -------------------------------------------------------------------------
if opts.display && (epoch == 1 || ~mod(epoch,opts.display))
    clf(fh);
    [nHidden1,~,nLabels] = size(W1); nVisibleAll = size(pvTrain,2);
    % Reconstructions train set ---------------------------------------
    tmp = cat(1,opts.dataTrain,pvTrain); ex = zeros(size(tmp));
    ex(1:2:end,:,:) = tmp(1:size(tmp,1)/2,:,:);
    ex(2:2:end,:,:) = tmp(size(tmp,1)/2+1:end,:,:);
    ex = reshape(reshape(ex,[],nLabels)*jet(nLabels),[],nVisibleAll,3);
    ex = permute(ex,[2 3 1]);
    ex = reshape(ex, opts.imageSize(1),opts.imageSize(2),size(ex,2),size(ex,3));
    subplot(221); montage(ex);
    title(['Estimated log-likelihood (train): ' num2str(info.llTrain(epoch))])
    % Reconstructions val set -----------------------------------------
    tmp = cat(1,opts.dataVal,pvVal); ex = zeros(size(tmp));
    ex(1:2:end,:,:) = tmp(1:size(tmp,1)/2,:,:);
    ex(2:2:end,:,:) = tmp(size(tmp,1)/2+1:end,:,:);
    ex = reshape(reshape(ex,[],nLabels)*jet(nLabels),[],nVisibleAll,3);
    ex = permute(ex,[2 3 1]);
    ex = reshape(ex, opts.imageSize(1),opts.imageSize(2),size(ex,2),size(ex,3));
    subplot(222); montage(ex);
    title(['Estimated log-likelihood (val): ' num2str(info.llVal(epoch))])
    % Filters (hidden-to-visible connections) -------------------------
    subplot(223);
    nFilters= nHidden1;
    filters = reshape(softmax(W1(1:nFilters,:,:)),[],nLabels) * jet(nLabels);
    if isempty(opts.grid)
        filters = permute(reshape(filters,[nFilters,opts.imageSize,3]),[2 3 4 1]);
    else
        filters = permute(reshape(filters,...
            [nFilters,opts.grid.patchSize,opts.grid.patchSize,3]),[2 3 4 1]);
    end
    montage(filters); title('Hidden-to-visible weights');
    % Error on train and validation sets ------------------------------
    subplot(224); hold on; axis square; grid on;
    plot(1:epoch, info.errorTrain, 'k');  plot(1:epoch, info.errorVal, 'r');
    xlabel('epoch'); ylabel('Reconstruction error'); legend('train', 'val');
    title(sprintf('Epoch %d, train error: %.4f, val error: %.4f', ...
        epoch, info.errorTrain(epoch),info.errorVal(epoch)));
    hold off; colormap jet; drawnow;
end

% -------------------------------------------------------------------------
function [pv, v, ph, h] = contrastiveDivergence(h, W, bv, bh, opts)
% -------------------------------------------------------------------------
for r=1:opts.cdRounds
    [pv,v] = rbmSampleVisible(h,W,bv,opts.grid,opts.pretrain);
    [ph,h] = rbmSampleHidden1(permute(v,[2,3,1]),W,bh,0,0,opts.grid,opts.pretrain);
end

% -------------------------------------------------------------------------
function [pv,v,ph1,h1,ph2,h2] = contrastiveDivergenceDeep(h1, W1, W2, bv, bh1, bh2, opts)
% -------------------------------------------------------------------------
for r=1:opts.cdRounds
    [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
    [pv,v]   = rbmSampleVisible(h1,W1,bv,opts.grid,opts.pretrain);
    [ph1,h1] = rbmSampleHidden1(permute(v,[2,3,1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
end

% -------------------------------------------------------------------------
function E = expectationShared(v, ph, opts)  
% -------------------------------------------------------------------------
nVisible = opts.grid.nVisiblePerPatch;
nHidden  = opts.grid.nHiddenPerPatch;
[~,nLabels,nExamples] = size(v);
E = zeros(nVisible*nLabels, nHidden);
for j=1:opts.grid.nPatches
    indVisible = opts.grid.patches(:,:,j); indVisible = indVisible(:);
	indHidden  = (1:opts.grid.nHiddenPerPatch) + (j-1)*opts.grid.nHiddenPerPatch;
    E = E + reshape(v(indVisible,:,:),nVisible*nLabels,nExamples) * ph(:,indHidden);
end
E = reshape(E',nHidden,nVisible,nLabels);

% -------------------------------------------------------------------------
function [ph1New, ph2New] = meanField(visible,W1,W2,bh1,bh2,opts)
% -------------------------------------------------------------------------
nExamples  = size(visible,3);
[nHidden, nVisible, nLabels] = size(W1);

inh1 = zeros(nExamples, opts.nHidden1); 
W1r  = reshape(W1,nHidden,nVisible*nLabels);
for j=1:opts.grid.nPatches
    indVisible = opts.grid.patches(:,:,j);  indVisible = indVisible(:);
    indHidden = (1:opts.grid.nHiddenPerPatch) + (j-1)*opts.grid.nHiddenPerPatch;
    inh1(:,indHidden) = (W1r * reshape(visible(indVisible,:,:),nnz(indVisible)*nLabels, []))';
end
ph1Old = sigmoid(inh1 + bh1);
ph2Old = sigmoid(ph1Old * W2 + bh2);

for i = 1:opts.nMeanFieldIterations % Number of the mean-field updates
    ph1New = sigmoid(inh1 + bh1 + ph2Old * W2');
    ph2New = sigmoid(ph1New * W2 + bh2);
    ph2Old = ph2New;
end

% -------------------------------------------------------------------------
function logLikelihood = computeLikelihood(pv,data,opts)
% -------------------------------------------------------------------------
logLikelihood = 0;
for label = 1:opts.nLabels
    logProb = log(pv(:,:,label));
    logLikelihood = logLikelihood + sum(logProb(data(:,:,label)==label));
end
logLikelihood = logLikelihood / size(data,1);

% -------------------------------------------------------------------------
function s = prob2state(p)
% -------------------------------------------------------------------------
s = p > rand(size(p));

% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
function e = freeEnergy(v,W,bv,bh,opts)
% -------------------------------------------------------------------------
[~,nVisibleAll,nLabels] = size(v);
x = bsxfun(@plus, inputFromVisible(v,W,opts.grid), bh);
e = -(reshape(v,[],nVisibleAll*nLabels)*bv(:) + sum(log(1+exp(x)),2));
