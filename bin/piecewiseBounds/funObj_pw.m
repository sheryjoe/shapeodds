function [f, gm, gv] = funObj_pw(m, v, bound)
% compute piecewise bound to E(log(1+exp(x))) where x~N(m,v)
% m and v can be vectors in which case outputs are vectors too.
% Written by Emtiyaz, CS, UBC
% Modified on Oct. 9, 2011

f  = zeros(size(m));
gm = zeros(size(m));
gv = zeros(size(v));
for r = 1:length(bound)
    l = bound(r).l;
    h = bound(r).h;
    a = bound(r).a(3);
    b = bound(r).a(2);
    c = bound(r).a(1);
    
    [fr, gmr, gvr]=objgradpart_vec(m,v,a,b,c,l,h);
    f  = f + fr;
    gm = gm + gmr;
    gv = gv + gvr;
end

