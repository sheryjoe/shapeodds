
function [shapeoddsOptions, rbmParams, rbmOptions] = InitializeShapeOddsUsingRBM(fsTrain, fdims, nworkers, nFolds, shapeoddsOptions, randnFactor_store, batchSize_store, outfilename)

[w0, W, rbmParams, rbmOptions] = InitializeWw0(fsTrain, shapeoddsOptions.L, fdims, nworkers, nFolds, randnFactor_store, batchSize_store, '');

shapeoddsOptions.w0_init_method = 'vector';
shapeoddsOptions.w0_init     = w0;
shapeoddsOptions.W_init_method  = 'matrix';
shapeoddsOptions.W_init      = W;

if exist('outfilename', 'var')
    save(outfilename, 'shapeoddsOptions', 'rbmParams', 'rbmOptions', '-v7.3');
end