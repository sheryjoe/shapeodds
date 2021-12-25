
function Wtilde_orig = UnsquareModelParameters2(Wtilde, fdims, fdims_orig, D_orig)

L = size(Wtilde,2);

for d = 1 : length(fdims_orig)
    idx{d} = [num2str(1) ':' num2str(fdims_orig(d))] ;
end

Wtilde_orig = zeros([D_orig L]);
for l = 1 : L
    wl              = Wtilde(:,l);
    wl              = reshape(wl, fdims);
    wl_orig         = zeros(fdims_orig);
    
    if length(fdims) == 2
        wl_orig(1:fdims_orig(1), 1:fdims_orig(2)) = wl(1:fdims_orig(1), 1:fdims_orig(2));
    else
        wl_orig(1:fdims_orig(1), 1:fdims_orig(2), 1:fdims_orig(3)) = wl(1:fdims_orig(1), 1:fdims_orig(2),1:fdims_orig(3));
    end
    %wl_orig(idx{:}) = wl(idx{:});
    Wtilde_orig(:,l)     = wl_orig(:);
end
