function a = GetNthMap(A,n)

idx(1:ndims(A) - 1) = {':'};
a = A(idx{:},n);