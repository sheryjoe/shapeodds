function A = SetNthMap(A,a,n)

idx(1:ndims(A) - 1) = {':'};
A(idx{:},n) = a;