
function epsi = epsi_function (t, tau, funcType)

% if ~exist('funcType' , 'var')
%     funcType = 'ModifiedBiancoYohai';
% end

% clamping just in case
if sum(t<0) > 0
    fprintf('Warning: negative t given to the rho function ... clamping will be performed ...!!!!! \n');
    t(t<0) = 0;
end

if tau < 0
    fprintf('Warning: negative tau given to the rho function ... clamping will be performed ...!!!!! \n');
    tau = 0;
end

switch funcType,
    case 'BiancoYohai',
        epsi = (1 - (t./tau)) .* (t <= tau) + (zeros(size(t))) .* (t > tau);
    
    case 'ModifiedBiancoYohai',
        st   = sqrt(t);
        stau = sqrt(tau);
        epsi  = (exp(-stau)) .* (t <= tau) + (exp(-st)) .* (t > tau);
end

