
function rho = rho_function (t, tau, funcType)

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
        rho = (t - (t.^2)/(2.*tau)) .* (t <= tau) + (ones(size(t)) .* (tau/2)) .* (t > tau);
    
    case 'ModifiedBiancoYohai',
        st   = sqrt(t);
        stau = sqrt(tau);
        rho  = (t.*exp(-stau)) .* (t <= tau) + (-2 .* exp(-st) .* (1 + st) + exp(-stau) .* (2 .*(1 + stau) + tau)) .* (t > tau);
end

