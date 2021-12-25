function imageNames = getImageNames(vocPath, class, set, vocYear)
% getImageNames  Get file names for PASCAL images
%   
%   INPUT:
%   vocPath: path for the VOCDevkit
%   class: string or cell array of strings with valid Pascal object
%       class names. Example: if class = {'bus', 'car'}, then this
%       function returns filenames for all the images that contain either a
%       car or a bus. If class == 'all', getImageNames returns all images.
%   set: one of 'train', 'test', 'val' {'train'}
%   vocYear: get images from all years <= vocYear

if nargin < 4, vocYear = '2010'; end
if nargin < 3, set = 'train'; end
if nargin < 2, class = 'all'; end

assert(ismember(set, {'train', 'trainval', 'test', 'val'}), ...
    'Invalid image set. Set must be one of train, trainval, test.')
assert(ischar(vocYear) && str2double(vocYear) <= 2012 && str2double(vocYear) >= 2007,...
    'vocYear must be a string from 2007-2012')
assert(ischar(class) || iscell(class),....
    'class must be a string or a cell array of strings')
assert(all(ismember(class, PascalObject.validObjectClasses())) ||...
    strcmp(class,'all'), 'Invalid object class')

if strcmp(class, 'all') % get images containing objects from all classes
    fid = fopen(fullfile(vocPath,'ImageSets', 'Main', [set '.txt']));
    assert(fid>0, 'Could not open file');
    imageList = textscan(fid,'%s %d');
    fclose(fid);
    imageNames = imageList{1};
    years  = str2double(cellfun(@(x) x(1:4), imageNames, 'UniformOutput', false));
    imageNames = imageNames(years <= str2double(vocYear));
else    % class is either a cell array or a string (single class)
    if ~iscell(class), class = {class}; end
    imageNames = cell(numel(class), 1);
    for i=1:numel(class)
        fid = fopen(fullfile(vocPath,'ImageSets', 'Main', [class{i} '_' set '.txt']));
        assert(fid>0, 'Could not open file');
        imageList = textscan(fid,'%s %d');
        fclose(fid);
        
        % Get positive ids for VOC2010 dataset (we use VOC2012, which is a 
        % superset of VOC2010, so some of the part annotations are missing)
        labels = imageList{2};
        imageNames{i} = imageList{1}(labels > 0);  % keep only positives
        years  = str2double(cellfun(@(x) x(1:4), imageNames{i}, 'UniformOutput', false));
        imageNames{i} = imageNames{i}(years <= str2double(vocYear));
    end    
    imageNames = unique(cat(1, imageNames{:}));
end

