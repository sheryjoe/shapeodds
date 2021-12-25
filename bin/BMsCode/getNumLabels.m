function n = getNumLabels(objectClass)
switch objectClass
    case 'circles'
        n = 1;
    case 'faces'
        n = 3;
    case {'cow','bird'}
        n = 5;
    case {'car','horse','aeroplaneOID'}
        n = 6;
    case {'person','pedestrian'}
        n = 7;
    otherwise
        error('Object class not supported')
end