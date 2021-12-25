function [input_for_hidden,hPos,posterior] = rbmChoosePosition(input_for_hidden,h2in,bh,grid,c)
if ~exist('c','var')
	c= 1;
end
for k = 1:grid.nTransl
	input_total_					= c*input_for_hidden(:,:,k) + h2in + bh;
	log_probability(:,k)	= sum(log(exp(input_total_) + 1),2);
end
probability = exp(bsxfun(@plus,log_probability,-max(log_probability,[],2)));
posterior = bsxfun(@times,probability,1./sum(probability,2));
hPos      =  drawCategorical_2D(posterior);
for k=1:grid.nTransl
	wt											= find(hPos(:,k));
	input_for_hidden_(wt,:) = input_for_hidden(wt,:,k);
end
input_for_hidden = input_for_hidden_;
