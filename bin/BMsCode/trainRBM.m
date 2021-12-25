%% Load segmentations
objectClass = 'faces';
[segsTrain,imagesTrain,cnnScoresTrain] = loadSegs(objectClass,'train',1,100);
[segsVal,imagesVal,cnnScoresVal]       = loadSegs(objectClass,'val');

%% Opts for different configurations
% opts = bmOpts(objectClass,'batchSize',10,'nEpochs',1000,'doCNN',0);
optsRBM = bmOpts(objectClass,'batchSize',30,'nEpochs',1000,'display',10,'doCNN',0,...
    'cdRounds',1,'PCD',1,'nHidden1',300,'saveStep',100,'W_0',0.1,'epsw_0',1e-4);
optsDBM = bmOpts(objectClass,'batchSize',10,'nEpochs',1000,'cdRounds',5,...
    'doDeep',1,'nHidden1',300,'nHidden2',50,'display',10,'saveStep',100);
optsRBM1 = bmOpts(objectClass,'batchSize',10,'nEpochs',1000,'display',10,...
    'gridSize',[2 2],'nHiddenPerPatch',100,'pretrain',1,'saveStep',100);
optsRBM2 = bmOpts(objectClass,'batchSize',10,'nEpochs',1000,'display',0,'nHidden1',50,'pretrain',2,'saveStep',100,'cdRounds',5);
optsSBM = bmOpts(objectClass,'batchSize',30,'nEpochs',1000,'gridSize',optsRBM1.gridSize,...
    'doDeep',1,'nHidden2',optsRBM2.nHidden1,'nHiddenPerPatch',optsRBM1.nHiddenPerPatch,...
    'display',10,'saveStep',100,'cdRounds',5,'epsw_0',1e-4);

indPerm = randperm(size(segsVal,1),30); % use random subset for evaluation
optsRBM.cnnScoresTrain = cnnScoresTrain;
optsRBM.dataVal = segsVal(indPerm,:,:);
optsRBM1.cnnScoresTrain = cnnScoresTrain;
optsRBM1.dataVal = segsVal(indPerm,:,:);
optsDBM.cnnScoresTrain = cnnScoresTrain;
optsDBM.dataVal = segsVal(indPerm,:,:);
optsSBM.cnnScoresTrain = cnnScoresTrain;
optsSBM.dataVal = segsVal(indPerm,:,:);
if ~isempty(cnnScoresVal)
    optsRBM.cnnScoresVal = cnnScoresVal(indPerm,:,:);
    optsRBM1.cnnScoresVal = cnnScoresVal(indPerm,:,:);
    optsDBM.cnnScoresVal = cnnScoresVal(indPerm,:,:);
    optsSBM.cnnScoresVal = cnnScoresVal(indPerm,:,:);
end
warning off

%% Train RBM/DBM/SBM
[infoRBM,Wcnn,W1,bv,bh1,W2,bh2] = rbm(segsTrain,optsRBM);


%% Train RBM/DBM/SBM (pre-train)

[infoRBM1,Wcnn,W1,bv,bh1]  = rbm(segsTrain,optsRBM1);
[ph1,h1] = rbmSampleHidden1(permute(segsTrain,[2 3 1]),W1,bh1,[],[],optsRBM1.grid,0);
[infoRBM2,Wcnn,W2,~,bh2] = rbm(h1,optsRBM2);
[infoSBM,Wcnn,W1,bv,bh1,W2,bh2] = rbm(segsTrain,optsSBM,[],W1,bv,bh1,W2',bh2);

