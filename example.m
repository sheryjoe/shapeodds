
%%
clc
clear all
close all
warning off

addpath(genpath('bin'));
addpath(genpath('bin/minFunc')); % avoid confusion with minFunc in UGM
% for reproducibility
rng('default');


%% input data

dataDir = 'data/';

fdimsFilename   = [dataDir 'horses_fdims.mat'];
fsTrainFilename = [dataDir 'horses_nTrain180_fsTrain.mat'];
fsTestFilename  = [dataDir 'horses_nTrain180_fsTest.mat'];

load(fdimsFilename, 'D', 'D_x', 'D_y', 'fdims');
load(fsTrainFilename, 'fsTrain');
load(fsTestFilename,  'fsTest');

% subsample for a quick test
% fsTrain = fsTrain(:,1:2:end);
% fsTest  = fsTest(:,1:2:end);

nTrain = size(fsTrain,2);
nTest  = size(fsTest,2);

nworkers = 8; % for parallel computation

%% optimization parameters
shapeoddsOptions  = FillInDefaultShapeOddsOptions(nTrain);

%% initialize shapeodds
randnFactor_store = [0.001 0.01 0.05];
batchSize_store   = round(nTrain/4);
nFolds            = 3;

[shapeoddsOptions, rbmParams, rbmOptions] = InitializeShapeOddsUsingRBM(fsTrain, fdims, nworkers, nFolds, shapeoddsOptions, randnFactor_store, batchSize_store);
%% train shapeodds
shapeoddsOptions.isFA     = 0;
shapeoddsOptions.ispar    = 1;
shapeoddsOptions.nworkers = nworkers;
shapeoddsOptions.out_prefix = 'models/horses_nTrain180/';

%shapeoddsOptions.NumberOfInits = 1; % for quick test but this needs to be > 3 or 5 for better init
[Z, Theta, Epsi, resStruct] = EstimateShapeOdds(UnVectorizeMaps(fsTrain, fdims, nTrain), shapeoddsOptions);

shapeoddsFilename = [shapeoddsOptions.out_prefix 'shapeodds.mat'];
save(shapeoddsFilename, 'Z','Theta','Epsi','resStruct','-v7.3');

%% visualize the model
PlotEpsi(Epsi);
VisTheta(Theta, D_x, D_y, 1);

%% infer unseen data to quantify generalization
alpha               = resStruct.alpha;
estep_options       = resStruct.estep_options;
estep_options.ispar = 1;

factorsPosterior      = Estep(fsTest, Theta, alpha, estep_options, []);
[~, phisTest, zTest]  = EvaluateShapeOdds(Theta, factorsPosterior, fsTest);

qsTest    = ComputeQ(phisTest);
VisImageSetWithOverlay2(fsTest, qsTest>=0.5, D_x, D_y, 'f (red) qmap > thresh (green), intersection  (yellow)', 1);

hamming_distance = mean(fsTest ~= (qsTest>=0.5),1);
cross_entropy    = -mean(fsTest .* log(qsTest+eps) + (1 - fsTest) .* log(1 - qsTest + eps), 1);

shapeoddsGenFilename = [shapeoddsOptions.out_prefix 'shapeodds_generalization.mat'];
save(shapeoddsGenFilename, 'factorsPosterior','zTest', 'qsTest','hamming_distance', 'cross_entropy','-v7.3');

%% robust inference

% generating corrupted label maps
contamination_rate    = 0.3;
corr_noise_sigma      = 2;
corruptionType        = 'background' ; % 'both', 'foreground', 'background'

fCs  = zeros(D,nTest);
fGTs = fsTest;

for n = 1 : nTest
    f_n = fGTs(:,n);
    
    [f_both, f_foreground, f_background] = AddCorrelatedNoise(UnVectorizeMaps(f_n, fdims,1), corr_noise_sigma, contamination_rate);
    switch corruptionType
        case 'both',
            fCs(:,n) = VectorizeMaps(f_both,D,1);
        case 'foreground',
            fCs(:,n) = VectorizeMaps(f_foreground,D,1);
        case 'background',
            fCs(:,n) = VectorizeMaps(f_background,D,1);
    end
end

VisImageSet2(fCs, D_x, D_y, sprintf('%s: corrsigma = %1.1f, rate = %1.1f, ii = %d', corruptionType,corr_noise_sigma,contamination_rate,ii), 1); drawnow;


estep_options       = resStruct.estep_options;
estep_options.MaxRobustIter        = 20;
estep_options.TauContractionFactor = 0.95; % contraction factor for tau
estep_options.RhoFuncType          = 'ModifiedBiancoYohai';
estep_options.ispar                = 1;
estep_options.nworkers             = nworkers;
[factorsPosteriorRobust, gammaRobust, logLikelihoodRobust, outlier_mask, tau] = RobustEstep(fCs, Theta, alpha, estep_options, []);

Zs               = factorsPosteriorRobust.posteriorMean;
Phis             = bsxfun(@plus, Theta.W  * Zs  , Theta.w0);
Qs               = ComputeQ(Phis);
hamming_distance = mean(fGTs ~= (Qs>=0.5),1);
cross_entropy    = -mean(fGTs .* log(Qs+eps) + (1 - fGTs) .* log(1 - Qs + eps), 1);

VisImageSetWithOverlay2(fGTs, Qs>=0.5, D_x, D_y, 'fGTs (red) qmap > thresh (green), intersection  (yellow)', 1);

shapeoddsRobustFilename = [shapeoddsOptions.out_prefix 'shapeodds_robustness.mat'];
save(shapeoddsRobustFilename, 'estep_options','fCs', 'fGTs','factorsPosteriorRobust', 'Zs', 'Qs', 'hamming_distance','cross_entropy','-v7.3');

%% realism

nZSamples = 1000;

% sampling the model from the posterior of known samples (eg fsTest)
QZs = [];
for n = 1 : nTest
    Mu        = factorsPosterior.posteriorMean(:,n);
    Sigma     = factorsPosterior.posteriorCov(:,:,n);
    Zs        = mvnrnd(Mu, Sigma,round(nZSamples/nTest))';
    PhiZs     = bsxfun(@plus, Theta.W  * Zs  , Theta.w0);
    QZs       = [QZs ComputeQ(PhiZs)];
end

VisImageSet2(QZs, D_x, D_y, 'samples from posterior', 1); drawnow;


% sampling from the prior
Mu        = Theta.Mu;
Sigma     = Theta.Sigma;
Zs        = mvnrnd(Mu, Sigma, nZSamples)';
PhiZs     = bsxfun(@plus, Theta.W  * Zs  , Theta.w0);
QZs       = ComputeQ(PhiZs);

VisImageSet2(QZs, D_x, D_y, 'samples from prior', 1); drawnow;