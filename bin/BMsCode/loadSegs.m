function [dataTrainMulti,images,cnnScores] = loadSegs(objectClass,set,normFactor,nSubset)
% We assume that all the RGB images have the same size

if nargin < 3, normFactor = 1; end

paths = setPaths();
imageSize = [41 41]; % same as the dimensions of the last layer of the DCNN
nLabels = getNumLabels(objectClass);
switch objectClass
    case 'faces'
        switch set
            case 'train'
                fp = fopen(paths.faces.trainList,'r'); assert(fp>0);
            case 'val'
                fp = fopen(paths.faces.valList,'r'); assert(fp>0);
            case 'test'
                fp = fopen(paths.faces.testList,'r'); assert(fp>0);
            otherwise
                error('Invalid image set')
        end
        imageNames = textscan(fp,'%s %s'); fclose(fp);
        if nargin > 3
            imageNames{1} = imageNames{1}(1:min(nSubset,numel(imageNames{1})));
            imageNames{2} = imageNames{2}(1:min(nSubset,numel(imageNames{2})));
        end
        for i=1:numel(imageNames{2})
            imageNames{2}{i} = [repmat('0',[1,4-numel(imageNames{2}{i})]), imageNames{2}{i}];
        end
        segFiles = strcat(paths.faces.segs, filesep, imageNames{1}, '_', imageNames{2}, '.ppm');
        
        data = zeros([imageSize, 3, numel(segFiles)], 'uint8');
        for i=1:numel(segFiles)
            data(:,:,:,i) = imresize(imread(segFiles{i}), imageSize, 'nearest');
        end
        if strcmp(set,'train') % add flipped versions (we don't have  flipped images for faces)
            data = cat(4,data,flipdim(data,2));
        end
        data = data(:,:,[3 1 2],:);
        dataTrainMulti = data > 0;
        if nargout > 1 % Read images (we assume that all images have the same size)
            imageFiles = strcat(paths.faces.images, filesep, imageNames{1}, '_', imageNames{2}, '.jpg');
            nImageFiles = numel(imageFiles);
            images = cell(nImageFiles,1);
            for i=1:nImageFiles
                images{i} = imread(imageFiles{i});
            end
            if strcmp(set,'train')
                for i=nImageFiles+1:2*nImageFiles
                    images{i} = flipdim(images{i-nImageFiles},2);
                end
            end
        end
    case 'pedestrian'
        segNames = dir(fullfile(paths.humaneva, '*.png'));
        segFiles = strcat(paths.humaneva, filesep, {segNames(:).name}');
        switch set
            case 'train'
                segFiles = segFiles(1:600);
            case 'val'
                segFiles = segFiles(601:683);
            otherwise
                error([set ' set is not supported for class ' objectClass])
        end
        data = zeros([imageSize, numel(segFiles)],'uint8');
        for i=1:numel(segFiles)
            data(:,:,i) = imresize(imread(segFiles{i}), imageSize, 'nearest');
        end
        if strcmp(set,'train') % add flipped versions (we don't have  flipped images for faces)
            data = cat(3,data,flipdim(data,2));
        end
        tmp = data; data = zeros(size(tmp),'uint8');
        % merge parts
        data(tmp == 19) = 1;                            % hair
        data(tmp == 9)  = 2;                            % face
        data(tmp == 29) = 3;                            % upper clothes -> torso
        data(tmp == 49 | tmp == 59) = 4;                % arms
        data(tmp == 39 | tmp == 69 | tmp == 79) = 5;    % lower clothes, legs --> legs
        data(tmp == 89 | tmp == 99) = 6;                % feet/shoes
        %             data(tmp == 10) = 1;    % hair
        %             data(tmp == 20) = 2;    % face
        %             data(tmp == 30) = 3;    % upper clothes/torso
        %             data(tmp > 50 & tmp < 55) = 4;  % arms
        %             data(tmp == 40 | tmp == 61 | tmp == 62) = 5;  % legs/lower clothes
        %             data(tmp == 63 | tmp == 64) = 6;
        dataTrainMulti = false(imageSize(1),imageSize(2), nLabels, size(data,3));
        for i=1:nLabels
            dataTrainMulti(:,:,i,:) = data == i-1;
        end
    case {'cow','horse','bird','car','aeroplaneOID'}
        % maybe it will be better to use a "clean" subset of non-occluded,
        % non-truncated instances--> replace 'train' with 'train_subset'.
        % This however results in a much smaller dataset
        %dir_train = 'train';
        segNames = dir(fullfile(paths.segs, objectClass, set,'*.png'));
        segFiles = strcat(fullfile(paths.segs, objectClass, set), filesep, {segNames(:).name}');
        data = zeros([imageSize, numel(segFiles)],'uint8');
        
        for i=1:numel(segFiles)
            data(:,:,i) = imresize(imread(segFiles{i}), imageSize, 'nearest');
        end
        dataTrainMulti = false(imageSize(1),imageSize(2), nLabels, size(data,3));
        for i=1:nLabels
            dataTrainMulti(:,:,i,:) = data == i-1;
        end
        % TODO: CORRECT THIS, AS IN FACES CASE
        if nargout > 1  % Read images (we assume that all images have the same size)
            imageNames = dir(fullfile(paths.images, objectClass, set,'*.jpg'));
            imageFiles = strcat(fullfile(paths.segs, objectClass, set), filesep, {imageNames(:).name}');
            nImageFiles = numel(imageFiles);
            images = cell(2*nImageFiles,1);
            for i=1:nImageFiles
                images{i} = imread(imageFiles{i});
                images{i+nImageFiles} = flipdim(images{i},2);
            end
        end        
    otherwise
        error('Object class not supported')
end
dataTrainMulti = reshape(dataTrainMulti, prod(imageSize), nLabels, size(dataTrainMulti,4));
dataTrainMulti = permute(dataTrainMulti, [3 1 2]);  % nExamples x nVisible x nLabels

% Load features -----------------------------------------------------------
if nargout > 2
    featPath = fullfile(paths.features,objectClass, 'layer41x41x7');
    switch objectClass
        case 'faces'
            featureFiles = strcat(featPath, filesep, imageNames{1}, '_', imageNames{2}, '.mat');
        case {'cow','horse','bird','car'}
            for i=1:numel(segNames)
                segNames(i).name(end-3:end) = '.mat';
            end
            featureFiles = strcat(featPath, filesep, {segNames(:).name}');
        case 'pedestrian'
            warning('The HumanEva dataset contains only segmentations, CNN features are not available')
            cnnScores = [];
            return
        otherwise
            error('Object class not supported')
    end
    assert(numel(segFiles) == numel(featureFiles))
    featDim = nLabels;
    cnnScores = zeros([numel(featureFiles), imageSize, featDim],'single');
    for i=1:numel(featureFiles)
        tmp = load(featureFiles{i}); cnnScores(i,:,:,:) = tmp.cnnScores;
    end
    if strcmp(objectClass,'faces') && strcmp(set,'train') % flip features for faces
        cnnScores = cat(1,cnnScores, flipdim(cnnScores,3));
    end
    cnnScores = reshape(cnnScores, [size(cnnScores,1),prod(imageSize),featDim]);
    cnnScores = cnnScores/normFactor;
end


% function [dataTrainMulti,cnnScores,wt,rejected] = loadSegs(objectClass,set,normFactor)
%
% if nargin < 3, normFactor = 1; end
%
% paths = setPaths();
% imageSize = [41 41]; % same as the dimensions of the last layer of the DCNN
% is_pascal = 0;
% switch objectClass
% 	case 'faces'
%         switch set
%             case 'train'
%                 fp = fopen(paths.faces.trainList,'r'); assert(fp>0);
%             case 'val'
%                 fp = fopen(paths.faces.valList,'r'); assert(fp>0);
%             case 'test'
%                 fp = fopen(paths.faces.testList,'r'); assert(fp>0);
%             otherwise
%                 error('Invalid image set')
%         end
% 		imageNames = textscan(fp,'%s %s'); fclose(fp);
% 		for i=1:numel(imageNames{2})
% 			imageNames{2}{i} = [repmat('0',[1,4-numel(imageNames{2}{i})]), imageNames{2}{i}];
% 		end
% 		segFiles = strcat(paths.faces.segs, filesep, imageNames{1}, '_', imageNames{2}, '.ppm');
%
% 		data = zeros([imageSize, 3, numel(segFiles)], 'uint8');
% 		for i=1:numel(segFiles)
% 			data(:,:,:,i) = imresize(imread(segFiles{i}), imageSize, 'nearest');
% 		end
% 		nLabels = 3;
% 		data = data(:,:,[3 1 2],:);
% 		dataTrainMulti = data > 0;
% 	case 'pedestrian'
% 		segNames = dir(fullfile(paths.humaneva, '*.png'));
% 		segFiles = strcat(paths.humaneva, filesep, {segNames(:).name}');
% 		segFiles = segFiles(1:683);
%
% 		data = zeros([imageSize, numel(segFiles)],'uint8');
% 		for i=1:numel(segFiles)
% 			data(:,:,i) = imresize(imread(segFiles{i}), imageSize, 'nearest');
% 		end
% 		% merge parts
% 		tmp = data;
% 		data = zeros(size(tmp),'uint8');
% 		data(tmp == 19) = 1;                            % hair
% 		data(tmp == 9)  = 2;                            % face
% 		data(tmp == 29) = 3;                            % upper clothes -> torso
% 		data(tmp == 49 | tmp == 59) = 4;                % arms
% 		data(tmp == 39 | tmp == 69 | tmp == 79) = 5;    % lower clothes, legs --> legs
% 		data(tmp == 89 | tmp == 99) = 6;                % feet/shoes
% 		nLabels   = 7;
% 		dataTrainMulti = false(imageSize(1),imageSize(2), nLabels, size(data,3));
% 		for i=1:nLabels
% 			dataTrainMulti(:,:,i,:) = data == i-1;
% 		end
% 	case {'cow','horse','bird','car','aeroplaneOID'}
% 		% maybe it will be better to use a "clean" subset of non-occluded,
% 		% non-truncated instances--> replace 'train' with 'train_subset'.
% 		% This however results in a much smaller dataset
% 		%dir_train = 'train';
% 		is_pascal = 1;
%         segNames = dir(fullfile(paths.segs, objectClass, set,'*.png'));
%         segFiles = strcat(fullfile(paths.segs, objectClass, set), filesep, {segNames(:).name}');
% 		data = zeros([imageSize, numel(segFiles)],'uint8');
%
%         for i=1:numel(segFiles)
%             data(:,:,i) = imresize(imread(segFiles{i}), imageSize, 'nearest');
%         end
%         switch objectClass
%             case {'horse','car','aeroplaneOID'}
%                 nLabels = 6;
%             case {'cow','bird'}
%                 nLabels = 5;
%             otherwise
%                 error('Check your object class')
%         end
% 		dataTrainMulti = false(imageSize(1),imageSize(2), nLabels, size(data,3));
% 		for i=1:nLabels
% 			dataTrainMulti(:,:,i,:) = data == i-1;
% 		end
% 	otherwise
% 		error('Object class not supported')
% end
% dataTrainMulti = reshape(dataTrainMulti, prod(imageSize), nLabels, size(dataTrainMulti,4));
% dataTrainMulti = permute(dataTrainMulti, [3 1 2]);  % nExamples x nVisible x nLabels
%
%
%
% for k=1:size(dataTrainMulti,1)
% 	t     = bwconncomp(~reshape(dataTrainMulti(k,:,1),[41,41]));
% 	nc(k) = t.NumObjects;
% 	[m,i] = max(squeeze(dataTrainMulti(k,:,:)),[],2);
% 	n_un(k) = length(unique(i));
% end
%
% if is_pascal
% 	wt = find((double(nc<=1).*double(n_un>=nLabels-1)));
% 	if strcmp(objectClass,'horse')
% 		if strcmp(set,'val_subset')
% 			wt([14,15,33,36,41,44,45,54,60,61]) = [];
%         else %if latent
% % 			wt = wt(1:2:end);
% % 			wt([3,7,12,14,25,30,34,58,64,68]) =[];
% 			wt([5,6,13,14,23,24,27,28,49,50,59,60,67,68,115,116,127,128,135,136]) =[];
% % 			wt([5,6,17,18,23,24,27:42,53,54,85:90,93,94,97,98,111,112,139:142,...
% %                 167,168,179,180,193:198]) = [];
% 		end
% 	end
% 	if strcmp(objectClass,'cow')
% 		if strcmp(set,'val_subset')
% 			wt([5,10,20,25,26,36,49,50,56,57,60,61,66,74]) = [];
% 		elseif (strcmp(set,'train_subset'))
% 			wt([21,22,33:38,45,46,49,50,63,64,67,68,83,84,85,86,91,92,99,100,103,104,127,128]) = [];
% 		end
% 	end
% else
% 	wt = nc<=1;
% end
% rejected = dataTrainMulti(setdiff(1:size(dataTrainMulti,1), wt),:,:);
%
% %if (is_pascal)&
% %	wt(1+[1,4,16,22,34,37,40,45,46,49,50,59,65,66,67]) = [];
% %end
% dataTrainMulti = dataTrainMulti(wt,:,:);
%
% % Load features -----------------------------------------------------------
% if nargout > 1
% 	featPath = fullfile(paths.features,objectClass, 'layer41x41x7');
% 	switch objectClass
% 		case 'faces'
% 			featureFiles = strcat(featPath, filesep, imageNames{1}, '_', imageNames{2}, '.mat');
% 		case {'cow','horse','bird','car'}
% 			for i=1:numel(segNames)
% 				segNames(i).name(end-3:end) = '.mat';
% 			end
% 			featureFiles = strcat(featPath, filesep, {segNames(:).name}');
% 		case 'pedestrian'
% 			disp('The HumanEva dataset contains only segmentations')
% 			disp('Cnn features are not available')
% 		otherwise
% 			error('Object class not supported')
% 	end
% 	assert(numel(segFiles) == numel(featureFiles))
% 	featDim = nLabels;
% 	cnnScores = zeros(numel(featureFiles),  prod(imageSize), featDim,'single');
%     for i=1:numel(featureFiles)
%         tmp = load(featureFiles{i});  sz = size(tmp.cnnScores);
%         cnnScores(i,:,:) = reshape(tmp.cnnScores, sz(1)*sz(2),sz(3));
%     end
%     cnnScores = cnnScores(wt,:,:)/normFactor;
% end
%
%
%


