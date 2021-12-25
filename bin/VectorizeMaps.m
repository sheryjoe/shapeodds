function Vecs = VectorizeMaps(Maps, nPixels, nSamples)

if nSamples == 1
    Vecs = Maps(:);
else
    Vecs  = zeros(nPixels,nSamples);
    for n = 1 : nSamples
        curMap    = GetNthMap(Maps, n);
        Vecs(:,n) = curMap(:);
    end
end