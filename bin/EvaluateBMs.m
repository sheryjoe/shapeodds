
function [qsTrain, fsTrainRec, qsTest, fsTestRec, errorTrain, errorTest] = EvaluateBMs(opts, fsTrain, fsTest, W1, bv, bh1, W2, bh2)


nTrain      = size(fsTrain,2);

if ~isempty(fsTest)
    nTest       = size(fsTest,2);
else
    qsTest = []; fsTestRec = []; errorTest = [];
end

D           = size(fsTrain,1);

% pack label maps in the format that bm code accept
fsTrainMaps = zeros(nTrain, D,1);
for ii = 1 : nTrain
    fsTrainMaps(ii,:,1) = fsTrain(:,ii);
end

if ~isempty(fsTest)
    fsTestMaps = zeros(nTest, D,1);
    for ii = 1 : nTest
        fsTestMaps(ii,:,1) = fsTest(:,ii);
    end
end

bvt = repmat(bv, [nTrain,  1]);
if ~isempty(fsTest)
    bvv = repmat(bv, [nTest,   1]);
end

if opts.doDeep  % Maybe use CD or gibbs Sampling code for evaluation
    
    % Val reconstructions
    h2 = rand(1,opts.nHidden2) > 0.5;
    if ~isempty(fsTest)
        [ph1,h1] = rbmSampleHidden1(permute(fsTestMaps,[2 3 1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
        [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
        [ph1,h1] = rbmSampleHidden1(permute(fsTestMaps,[2,3,1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
        [pvv,vv] = rbmSampleVisible(h1,W1,bvv,opts.grid,opts.pretrain);
    end
    
    % Train reconstructions
    h2 = rand(1,opts.nHidden2) > 0.5;
    [ph1,h1] = rbmSampleHidden1(permute(fsTrainMaps,[2 3 1]) ,W1,bh1,h2,W2,opts.grid,opts.pretrain);
    [ph2,h2] = rbmSampleHidden2(h1,W2,bh2);
    [ph1,h1] = rbmSampleHidden1(permute(fsTrainMaps,[2 3 1]),W1,bh1,h2,W2,opts.grid,opts.pretrain);
    [pvt,vt] = rbmSampleVisible(h1,W1,bvt,opts.grid,opts.pretrain);
    
    qsTrain    = pvt;
    fsTrainRec = vt;
else
    if ~isempty(fsTest)
        % Val reconstructions
        [ph,h]   = rbmSampleHidden1(permute(fsTestMaps,[2 3 1]),W1,bh1,0,0,opts.grid,opts.pretrain);
        [pvv,vv] = rbmSampleVisible(h,W1,bvv,opts.grid,opts.pretrain);
    end
    
    % Train reconstructions
    [ph,h]   = rbmSampleHidden1(permute(fsTrainMaps,[2 3 1]),W1,bh1,0,0,opts.grid,opts.pretrain);
    [pvt,vt] = rbmSampleVisible(h,W1,bvt,opts.grid,opts.pretrain);
end

if ~isempty(fsTest)
    qsTest     = pvv';
    fsTestRec  = vv';
end
qsTrain    = pvt';
fsTrainRec = vt';

if ~isempty(fsTest)
    errorTest  = sum((fsTestMaps(:)-vv(:)).^2) / nTest;
end
errorTrain = sum((fsTrainMaps(:)-vt(:)).^2) / nTrain;
