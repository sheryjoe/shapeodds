function opts = bmOpts(objectClass, varargin)

% Update with values given by the user
for i = 1:2:length(varargin)
    key = varargin{i};
    val = varargin{i+1};
    opts.(key) = val;
end

% General training settings
opts = cv(opts,'verbose_text',objectClass);

opts = cv(opts,'imageSize',[41,41]);
opts = cv(opts,'batchSize',inf);
opts = cv(opts,'momentum',0.5);
opts = cv(opts,'pretrain', 0);
opts = cv(opts,'nEpochs',50);
opts = cv(opts,'display',1);    % display results every opts.display epochs
opts = cv(opts,'displayHiddenActivations',20);
opts = cv(opts,'saveStep',0);
opts = cv(opts,'cdRounds',1);
opts = cv(opts,'PCD',0);
opts = cv(opts,'verbose',1);

% Number of hidden units
opts = cv(opts,'nHidden1',100);
opts = cv(opts,'nHidden2',0);
opts = cv(opts,'nHiddenPerPatch',100);
opts = cv(opts,'overlap', 5);
opts = cv(opts,'gridSize',[1 1]);

% Learning rates
opts = cv(opts,'weightCost', 2e-4);
opts = cv(opts,'nMeanFieldIterations',10);
opts = cv(opts,'epochChangeMomentum',5);
opts = cv(opts,'W_0',0.1);
opts = cv(opts,'wDecay',2);
opts = cv(opts,'wDecayStep',200); % set to inf to freeze the learning rate
opts = cv(opts,'epsw_0',0.01);
opts = cv(opts,'epsw', opts.epsw_0 * ones(1,opts.nEpochs) ./ ...
    opts.wDecay.^floor((1:opts.nEpochs)/opts.wDecayStep));
opts = cv(opts,'epsbv',  opts.epsw);
opts = cv(opts,'epsbh',  opts.epsw);
opts = cv(opts,'epswcnn',opts.epsw);

% Flags
opts = cv(opts,'doCNN',0);
opts = cv(opts,'doDeep',0);
opts = cv(opts,'normFactor',10);
opts.useGrid = any(opts.gridSize > 1);
% Grid parameters
if opts.useGrid
    grid.shareWeights = true;				% share weights across grid patches
    grid.overlap      = opts.overlap;		% overlap between neighboring patches
    grid.size         = opts.gridSize;
    grid.nPatches     = prod(grid.size);
    grid.imageSize    = opts.imageSize;
    grid			  = getGridPatches(grid);
    
    grid.nVisiblePerPatch = grid.patchSize^2;
    grid.nHiddenPerPatch  = opts.nHiddenPerPatch;
    opts.nHidden1         = grid.nPatches * grid.nHiddenPerPatch;
    opts.grid             = grid;
    opts.gridString       = sprintf('_gridSize_%i_%i_overlap_%i',...
        opts.gridSize(1),opts.gridSize(2),opts.overlap);
else
    opts.grid = [];
    opts.gridString = '';
end

opts = cv(opts,'objectClass',objectClass);
opts = cv(opts,'nLabels',getNumLabels(objectClass));


suffix = sprintf(['rbm_%s_nHid1_%i_nHid2_%i' opts.gridString '_doCNN_%i_pretrain_%i',...
    '_CD%i_PCD%i_epsw_0_%.4f_W_0_%.3f_batchSize_%i_nEpochs_%i.mat'],...
    opts.objectClass,opts.nHidden1,opts.nHidden2,opts.doCNN,opts.pretrain,...
    opts.cdRounds,opts.PCD,opts.epsw_0,opts.W_0,opts.batchSize,opts.nEpochs);
opts.saveFile = ['sbm/',suffix];


function opts = cv(opts, key, val)
if ~isfield(opts,key)
    opts.(key) = val;
end




% function opts = bmOpts(varargin)
% 
% for i = 1:2:length(varargin)
%     key = varargin{i};
%     val = varargin{i+1};
%     opts.(key) = val;
% end
% 
% opts = cv(opts,'batchSize',inf);
% opts = cv(opts,'pretrain',0);
% opts = cv(opts,'nEpochs',50);
% opts = cv(opts,'cdRounds',10);
% opts = cv(opts,'nHidden1',128);
% opts = cv(opts,'weightCost',2e-4);
% opts = cv(opts,'smoothCost',0);
% opts = cv(opts,'momentum',0);
% opts = cv(opts,'display',0);
% opts = cv(opts,'do_cnn',1);
% opts = cv(opts,'do_DCD',0);
% opts = cv(opts,'W_0',2);
% 
% opts = cv(opts,'useBiasVisible',0);
% opts = cv(opts,'useBiasHidden',	1);
% opts = cv(opts,'nMeanFieldIterations',10);
% opts = cv(opts,'epochChangeMomentum',5);
% opts = cv(opts,'epswFactor',.01);
% opts = cv(opts,'saveStep',10);
% opts = cv(opts,'epsw',opts.epswFactor * ones(1,opts.nEpochs) ./ ceil((1:opts.nEpochs)/(0.1*opts.nEpochs)));
% opts = cv(opts,'epsbv',opts.epsw);
% opts = cv(opts,'epsbh',opts.epsw);
% opts = cv(opts,'epswcnn',opts.epsw);
% opts = cv(opts,'imageSize',[41,41]);
% opts = cv(opts,'nHiddenPerPatch',100);
% opts = cv(opts,'use_grid',1);
% opts = cv(opts,'use_epitome',false);
% opts = cv(opts,'nHidden2',0);
% opts = cv(opts,'overlap',5);
% opts = cv(opts,'size',1);
% opts = cv(opts,'norm_factor',10);
% opts = cv(opts,'doLatent',0);
% opts = cv(opts,'fct_w',1);
% 
% % Grid parameters
% if opts.use_grid
%     grid.shareWeights = true;								% share weights across grid patches
%     grid.overlap      = opts.overlap;				% overlap between neighboring patches
%     grid.size         = [opts.size opts.size];
%     grid.nPatches     = prod(grid.size);
%     grid.use_epitome  = opts.use_epitome;
%     grid							= getGridPatches(opts.imageSize, grid);
%     %[grid.epitome] = getGridPatches(opts.imageSize, grid.size, grid.overlap);
%     
%     grid.nPatches					= prod(grid.size);
%     grid.nVisiblePerPatch = grid.patchSize^2;
%     grid.nHiddenPerPatch = opts.nHiddenPerPatch;
%     opts					= cv(opts,'grid',grid);
%     opts.nHidden1 = opts.grid.nPatches * opts.grid.nHiddenPerPatch;
%     opts.grd_str = sprintf('size_%i_overlap_%i',opts.size,opts.overlap);
% else
%     opts = cv(opts,'grid',[]);
%     opts.grd_str = '';
% end
% 
% % suffix = sprintf('rbm_%s_nhd1_%i_nhd2_%i_%s_nEpochs_%i_W0_%.1f_latent_%i.mat',...
% % 	opts.objectClass,opts.nHidden1,opts.nHidden2,opts.grd_str,...
% % 	opts.nEpochs,opts.W_0,opts.doLatent);
% suffix = sprintf('rbm_%s_nhd1_%i_weightCost_%.2f_W_0_%.1f_fct_w_%.2f_nEpochs_%i_latent_%i.mat',...
%     opts.objectClass,opts.nHidden1,opts.weightCost,opts.W_0,opts.fct_w,...
%     opts.nEpochs,opts.doLatent);
% opts.saveFile = ['sbm/',suffix];
% if opts.do_DCD
%     opts.saveFile = strrep(opts.saveFile,'.mat','_DCD.mat');
% end
% 
% if ~opts.useBiasVisible
%     opts.epsbv  = 0*opts.epsbv;
% end
% 
% if ~opts.useBiasHidden
%     opts.epsbh  = 0*opts.epsbh;
% end
% 
% opts = cv(opts,'nLabels',getNumLabels(opts.objectClass));
% 
% 
% function opts = cv(opts, key, val)
% if ~isfield(opts,key)
%     opts.(key) = val;
% end

