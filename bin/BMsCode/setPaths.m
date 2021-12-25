function paths = setPaths(is_iasonas)
% SETPATHS Set default paths according to your directory structure

% VOC2012
paths.VOC2012.devkit    = '~/datasets/VOC2012/VOCdevkit';
paths.VOC2012.data      = fullfile(paths.VOC2012.devkit, 'VOC2012');
paths.VOC2012.images    = fullfile(paths.VOC2012.data, 'JPEGImages');
paths.VOC2012.annos     = fullfile(paths.VOC2012.data, 'Annotations');
paths.VOC2012.code      = fullfile(paths.VOC2012.devkit, 'VOCcode');

% PennFudan
paths.pennfudan.root    = '~/datasets/pedestrian_parsing';
paths.pennfudan.images  = fullfile(paths.pennfudan.root, 'Color');
paths.pennfudan.segs    = fullfile(paths.pennfudan.root, 'GroundTruth');
paths.pennfudan.fowlkes = fullfile(paths.pennfudan.root, 'Our_Results');
paths.humaneva          = fullfile(paths.pennfudan.root, 'train_groundtruth');

% ETHZ-cars
paths.ethzcars.images   = '~/datasets/ETHZ-cars/Thomas-cars/cars';
paths.ethzcars.segs     = '~/datasets/ETHZ-cars/Thomas-cars/parts';

% Faces in the Wild
paths.faces.root        = '~/datasets/FacesInTheWild';
paths.faces.images      = fullfile(paths.faces.root, 'images');
paths.faces.segs        = fullfile(paths.faces.root, 'segmentations');
paths.faces.trainList   = fullfile(paths.faces.root, 'parts_train.txt');
paths.faces.valList     = fullfile(paths.faces.root, 'parts_validation.txt');
paths.faces.testList    = fullfile(paths.faces.root, 'parts_test.txt');
paths.faces.partLabels  = fullfile(paths.faces.root, 'parts_lfw_funneled_gt');
paths.faces.superpixels = fullfile(paths.faces.root, 'parts_lfw_funneled_superpixels_mat');

% CUB200
paths.cub200.root       = '~/datasets/CUB_200_2011';
paths.cub200.images     = fullfile(paths.cub200.root, 'images');
paths.cub200.imageList  = fullfile(paths.cub200.root, 'images.txt');
paths.cub200.trainSplit = fullfile(paths.cub200.root, 'train_test_split.txt');
paths.cub200.trainData  = fullfile(paths.cub200.root, 'bird_train.mat');
paths.cub200.testData   = fullfile(paths.cub200.root, 'bird_test.mat');


% Pascal Parts and other directories
paths.annos             = 'PASCAL_PARTS/Annotations_Part/';
paths.protos            = 'protos';
paths.models            = 'models';
paths.groundtruth       = 'groundtruth';
paths.features          = 'features';
paths.lists             = 'lists';
paths.images            = fullfile(paths.groundtruth, 'partImages');
paths.segs              = fullfile(paths.groundtruth, 'partSegs');
