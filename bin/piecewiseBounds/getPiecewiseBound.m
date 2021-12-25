function bound = getPiecewiseBound(type,numOfPieces)
% for 'linear' type, numOfPieces should be <= 100
% for 'quad' type, it could be between 3 to 10 or 15,20.
% Written by Emtiyaz, CS, UBC
% Modified on Oct 9, 2011

switch type
    case 'linear'
        bound = getPieceWiseLinearBound(numOfPieces);
    case 'quad'
        bound = getbound(numOfPieces);
    otherwise
        error('no such type');
end

