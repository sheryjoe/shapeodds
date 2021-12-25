function [ph2,h2] = rbmSampleHidden2(h1, W2, bh2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    bh2 = 0;
elseif size(h1,1) ~= size(bh2,1)
    bh2 = repmat(bh2, [size(h1,1), 1]); 
end

ph2 = sigmoid(h1 * W2 + bh2);
if nargout > 1
    h2  = ph2 > rand(size(ph2));
end
