
function numhid = getNumberOfHiddenUnits(samples, labels)

% estimate the number of hidden units to be used in the hidden
% layer of rbm, it is related to the information content of the
% training samples and the number of training samples - refer to
% hinton's recipes paper

% I will use the chopps paper shape variability metric in order to
% encode the average information content per class, since it
% reflects the increase of shape variability as more random walk
% noise being added to the shapes, compared to just pixelwise
% entropy, 

% @inproceedings{li2013exploring,
%   title={Exploring Compositional High Order Pattern Potentials for Structured Output Learning},
%   author={Li, Yujia and Tarlow, Daniel and Zemel, Richard},
%   booktitle={Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on},
%   pages={49--56},
%   year={2013},
%   organization={IEEE}
% }


[D,N] = size(samples);
if isempty(labels)
    labels = ones(1,N);
end

classes      = unique(labels);
totalSamples = size(samples,2);
H_k   = [];
pi_k  = [];
for cid = 1 : length(classes)
    
    indices    = find(labels == classes(cid));
    curSamples = samples(:, indices);
    
    [nPixels, nSamples] = size(curSamples);
    
    % compute the fraction of cases for which the pixel i is on
    % (across all instances within the current class/cluster)
    q_k = sum(curSamples,2)./nSamples;
    %imshow(reshape(q_k,[181 181]), [])
    
    % compute the within-cluster average entropy
    %h_k = -(q_k .* log(q_k + eps) + (1-q_k) .* log(1-q_k + eps));
    h_k = -(q_k .* log2(q_k + eps) + (1-q_k) .* log2(1-q_k + eps));
    h_k(h_k < 0) = 0;
    %imshow(reshape(h_k,[181 181]), [])
    
    % compute the average entropy within the current cluster
    H_k(cid)  = sum(h_k)/nPixels;
    pi_k(cid) = nSamples / totalSamples;
end

% shape variability measure
V = sum(pi_k .* H_k);
nhidden = ceil(V * totalSamples);

% to be compatible with shapebm, give number of hidden units
% dividisble by four
numhid = ceil(nhidden/4)*4;