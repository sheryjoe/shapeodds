
function [Loglikelihood, phis, Z]  = EvaluateShapeOdds(Theta, factorsPosterior, fs)

Z     = factorsPosterior.posteriorMean;
phis  = bsxfun(@plus, Theta.W  * Z  , Theta.w0);
Loglikelihood  = ComputeSamplesLogLikelihood(fs, phis);

