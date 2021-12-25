
function [w0, W, rbmParams, rbmOptions] = InitializeWw0(fsTrain, L, fdims, nworkers, nFolds, randnFactor_store, batchSize_store, verbosetext, w0_init, W_init)

if ~exist('w0_init', 'var')
    w0_init = [];
end

if ~exist('W_init', 'var')
    W_init = [];
end

if ~exist('verbosetext', 'var')
    verbosetext = '';
end
if ~exist('randnFactor_store', 'var')
    randnFactor_store       = [0.001 0.01 0.1];
end

[D, nTrain] = size(fsTrain);
if ~exist('batchSize_store', 'var')
    batchSizeFactor_store   = [2 3 4 5 6 7 8 9 10];
    batchSize_store = ceil(ones(1, length(batchSizeFactor_store)).*nTrain./batchSizeFactor_store);
    batchSize_store = unique(batchSize_store);
end

if (length(batchSize_store) > 1) || (length(randnFactor_store) > 1)
    if nworkers > 0
        poolobj = gcp('nocreate');
        if isempty(poolobj)
            myCluster            = parcluster('local');
            myCluster.NumWorkers = nworkers;
            parpool('local', nworkers);
        end
    end
end

fsTrainShuffled = fsTrain(:, randperm(nTrain));

% rbm init 
objectClass = 'circles';

% pack label maps in the format that bm code accept
fsTrainMaps = zeros(nTrain, D,1); % examples x pixels x labels
for ii = 1:nTrain
    fsTrainMaps(ii,:,1) = fsTrainShuffled(:,ii);
end

pretrain    = 1;
nEpochs     = 3000;
nHiddenRBM  = L;

%pretrain    = 0;
%nEpochs     = 1000;
%nHiddenRBM  = L;

cdRounds    = 1;
display     = 0;

if (length(batchSize_store) > 1) || (length(randnFactor_store) > 1)
    
    %nFolds      = min(nTrain,3);
    
    % get the indices for each fold
    folds_indices = GetFoldsIndices(nTrain, nFolds);
    
    rbm_hamming_distance    = [];
    rbm_cross_entropy       = [];
    
    for bid = 1 : length(batchSize_store)
        batchSize = batchSize_store(bid);
        
        % use random subset for evaluation
        indPerm   = randperm(nTrain, batchSize);
        dataVal   = zeros(batchSize, D,1);
        for ii = 1 : batchSize
            dataVal(ii,:,1) = fsTrainShuffled(:,indPerm(ii));
        end
        
        for rid = 1 : length(randnFactor_store)
            
            randnFactor = randnFactor_store(rid);
            
            % Opts for different configurations
            rbmOptions = bmOpts(objectClass, 'imageSize', fdims, 'doDeep', 0, 'useGrid', 0, 'batchSize',batchSize, 'nEpochs',nEpochs, 'nHidden1',nHiddenRBM, 'display', display, ...
                'pretrain', pretrain, 'cdRounds',cdRounds, 'W_0', randnFactor, 'verbose',1);
            
            % use random subset for evaluation
            rbmOptions.dataVal = dataVal;
            rbmOptions.verbose_text = sprintf('%s: bid = %d of %d, rid = %d of %d', verbosetext, bid, length(batchSize_store), rid, length(randnFactor_store));
            
            parfor k = 1: nFolds
            %    for k = 1: nFolds
                
                [cur_fsTrain, cur_fsTest] = GetFoldTrainTestPartitions (fsTrainShuffled, k, folds_indices);
                cur_nTrain = size(cur_fsTrain,2);
                
                % pack label maps in the format that bm code accept
                cur_fsTrainMaps = zeros(cur_nTrain, D,1); % examples x pixels x labels
                for ii = 1:cur_nTrain
                    cur_fsTrainMaps(ii,:,1) = cur_fsTrain(:,ii);
                end
                
                % Train RBM
                rbmParams = [];
                if isempty(w0_init) || isempty(W_init)
                    [rbmParams.logs, rbmParams.Wcnn, rbmParams.W1, rbmParams.biasVisible, rbmParams.biasHidden1, rbmParams.W2, rbmParams.biasHidden2] = rbm(cur_fsTrainMaps,rbmOptions);
                else
                    [rbmParams.logs, rbmParams.Wcnn, rbmParams.W1, rbmParams.biasVisible, rbmParams.biasHidden1, rbmParams.W2, rbmParams.biasHidden2] =...
                        rbm(cur_fsTrainMaps,rbmOptions, [], W_init', w0_init', [],[],[]);
                end
                
                % infer unseen samples
                [qsTrain, ~, qsTest, ~] = EvaluateBMs(rbmOptions, cur_fsTrain, cur_fsTest, rbmParams.W1, rbmParams.biasVisible, rbmParams.biasHidden1, rbmParams.W2, rbmParams.biasHidden2);
                
                
                if 0
                    figure, subplot(2,3,1); VisImageSet2([rbmParams.biasVisible' rbmParams.W1'], D_x,D_y,'w0 W',1, gca);
                    subplot(2,3,2);VisImageSet2(cur_fsTrain, D_x, D_y, 'cur fsTrain', 1,gca); drawnow;
                    subplot(2,3,3);VisImageSet2(qsTrain, D_x, D_y, 'qsTrain', 1,gca); drawnow;
                    subplot(2,3,5); VisImageSet2(cur_fsTest, D_x, D_y, 'cur fsTest', 1,gca); drawnow;
                    subplot(2,3,6);VisImageSet2(qsTest, D_x, D_y, 'qsTest', 1,gca); drawnow;
                end
                
                %VisImageSet2(qsTrain, D_x, D_y, 'qsTrain', 1); drawnow;
                %VisImageSet2(qsTest, D_x, D_y, 'qsTest', 1); drawnow;
                
                rbm_hamming_distance(bid,rid,k) = mean(mean(cur_fsTest ~= (qsTest>=0.5),1));
                rbm_cross_entropy(bid,rid,k)    = mean(-mean(cur_fsTest .* log(qsTest+eps) + (1 - cur_fsTest) .* log(1 - qsTest + eps), 1));
            end
        end
    end
    
    rbm_hamming_distance = mean(rbm_hamming_distance,3);
    rbm_cross_entropy    = mean(rbm_cross_entropy,3);
    
    val = min(rbm_cross_entropy(:));
    for bid = 1 : length(batchSize_store)
        batchSize = batchSize_store(bid);
        for rid = 1 : length(randnFactor_store)
            randnFactor = randnFactor_store(rid);
            if rbm_cross_entropy(bid,rid) == val
                break
            end
        end
        if rbm_cross_entropy(bid,rid) == val
            break
        end
    end
    
else
    batchSize   = batchSize_store(1);
    randnFactor = randnFactor_store(1);
end

% use random subset for evaluation
indPerm   = randperm(nTrain, batchSize);
dataVal   = zeros(batchSize, D,1);
for ii = 1 : batchSize
    dataVal(ii,:,1) = fsTrainShuffled(:,indPerm(ii));
end

% [val, index] = min(rbm_cross_entropy);
% randnFactor = randnFactor_store(index);

% Opts for different configurations
rbmOptions = bmOpts(objectClass, 'imageSize', fdims, 'doDeep', 0, 'useGrid', 0, 'batchSize',batchSize, 'nEpochs',nEpochs, 'nHidden1',nHiddenRBM, 'display', display, ...
    'pretrain', pretrain, 'cdRounds',cdRounds, 'W_0', randnFactor, 'verbose',1);

% use random subset for evaluation
rbmOptions.dataVal = dataVal;

rbmOptions.verbose_text = sprintf('%s: batchSize = %d , randnFactor = %f', verbosetext, batchSize, randnFactor);
% Train RBM
rbmParams = [];
if isempty(w0_init) || isempty(W_init)
    [rbmParams.logs, rbmParams.Wcnn, rbmParams.W1, rbmParams.biasVisible, rbmParams.biasHidden1, rbmParams.W2, rbmParams.biasHidden2] = rbm(fsTrainMaps,rbmOptions);
else
    [rbmParams.logs, rbmParams.Wcnn, rbmParams.W1, rbmParams.biasVisible, rbmParams.biasHidden1, rbmParams.W2, rbmParams.biasHidden2] =...
        rbm(fsTrainMaps,rbmOptions, [], W_init', w0_init', [],[],[]);
end

% infer unseen samples
%[qsTrain] = EvaluateBMs(rbmOptions, fsTrain, [], rbmParams.W1, rbmParams.biasVisible, rbmParams.biasHidden1, rbmParams.W2, rbmParams.biasHidden2);

%VisImageSet2(qsTrain, D_x, D_y, 'qsTrain', 1); drawnow;
%VisImageSet2(qsTest, D_x, D_y, 'qsTest', 1); drawnow;

w0     = rbmParams.biasVisible';
W      = rbmParams.W1';
