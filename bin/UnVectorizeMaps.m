function Maps = UnVectorizeMaps(Vecs, sample_dims, nSamples)

if nSamples == 1
    Maps    = reshape(Vecs, sample_dims);
else
    Maps  = zeros([sample_dims, nSamples]);
    for n = 1 : nSamples
        curMap    = reshape(Vecs(:,n), sample_dims);
        Maps      = SetNthMap(Maps, curMap, n);
    end
end