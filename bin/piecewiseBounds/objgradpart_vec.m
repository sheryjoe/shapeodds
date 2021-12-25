function [fr,gm,gv]=objgradpart_vec(m,v,a,b,c,l,h)
% Compute expectation of a quadratic a*x^2 + b*x + c
% with respect to a trucated Gaussian with mean m and variance v with l<x<h
% See online appendix for detail

v = v + 1e-30;

D = size(m);

zl = (l-m)./sqrt(v);
zh = (h-m)./sqrt(v);

pl = normpdf(zl(:)')'./sqrt(v); %normal pdf
ph = normpdf(zh(:)')'./sqrt(v); %normal pdf
cl = 0.5*erf(zl/sqrt(2)); %normal cdf -const
ch = 0.5*erf(zh/sqrt(2)); %normal cdf -cosnt

if(h<l)
  error('Lower limit l must be less than upper limit h');
end

if(all(v<=0))
  error('Normal variance must be strictly positive');
end

%Compute trucated first and second moments
if(l==-inf & h==inf)
  %Compute moments and gradients with no truncation 
  ex0=1;
  ex1=m;
  ex2=v+m.^2;
  gm = bsxfun(@plus, 2*a.*m, b);
  gv = a;

else

  ex0 = ch-cl;

  %Compute truncated first moment
  ex1= v.*(pl-ph) + m.*(ch-cl);

  if(l==-inf)

    %Compute truncated second moment
    ex2=  v.*(0 - (h+m).*ph) + (v+m.^2).*(ch - cl);

    %Compute Gradient wrt to mean
    gm = a.*( 0 - (h^2+2*v).*ph) + a.*(2*m.*(ch-cl)); 
    gm = gm + b.*(0-h*ph) + b.*(ch-cl);
    gm = gm + c*(0-ph);

    %Compute Gradient wrt to variance
    gv = a/2./v.*( 0 - (2*v*h + h^3 -h^2*m).*ph) +a.*(ch-cl);
    gv = gv + b/2./v.*( 0 - (h^2+v-h*m).*ph); 
    gv = gv + c/2./v.*(0-(h-m).*ph);

  elseif(h==inf)

    %Compute truncated second moment
    ex2=  v.*((l+m).*pl - 0) + (v+m.^2).*(ch - cl);

    %Compute Gradient wrt to mean
    gm = a.*( (l^2+2*v).*pl - 0) + a.*2.*m.*(ch-cl); 
    gm = gm + b.*(l*pl-0) + b.*(ch-cl);
    gm = gm + c.*(pl-0); 

    %Compute Gradient wrt to variance
    gv = a/2./v.*( (2*v*l + l^3 -l^2*m).*pl - 0) +a.*(ch-cl);
    gv = gv + b/2./v.*( (l^2+v-l*m).*pl - 0); 
    gv = gv + c/2./v.*((l-m).*pl-0);

  else

    %Compute truncated second moment
    ex2=  v.*((l+m).*pl - (h+m).*ph) + (v+m.^2).*(ch - cl);

    %Compute Gradient wrt to mean
    gm = a.*( (l^2+2*v).*pl - (h^2+2*v).*ph) + a.*2.*m.*(ch-cl); 
    gm = gm + b.*(l*pl-h*ph) + b.*(ch-cl);
    gm = gm + c*(pl-ph);

    %Compute Gradient wrt to variance
    gv = a/2./v.*( (2*v*l + l^3 -l^2*m).*pl - (2*v*h + h^3 -h^2*m).*ph) +a.*(ch-cl);
    gv = gv + b/2./v.*( (l^2+v-l*m).*pl - (h^2+v-h*m).*ph); 
    gv = gv + c/2./v.*((l-m).*pl-(h-m).*ph);

  end
end

%Compute fr
fr = a.*ex2 + b.*ex1 + c.*ex0;

return



