
Content:
    - example.m: run a demo examples using Weizmann horses dataset. In this example you can learn a shapeodds model, perform inference and robust inference, quantify model generalization, and draw samples from the model.
    - EstimateShapeOdds.m: the main function that learns shapeodds model given a set of silhouttes
    - bin: includes all auxilary functions used by EstimateShapeOdds.m and example.m
    - data: a folder where data used in example.m is stored
    - models: a folder where you can store learned models and other outputs


For any questions, please contact Shireen Elhabian (shireen@sci.utah.edu)

If you use this code in any publication, please cite the following:

shapeodds:
Shireen Y. Elhabian and Ross T. Whitaker. ShapeOdds: Variational Bayesian Learning of Generative Shape Models. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2231-2242, 2017.

piecewise bounds:
Marlin BM, Khan ME, Murphy KP. Piecewise Bounds for Estimating Bernoulli-Logistic Latent Gaussian Models. InICML 2011 Jun 28 (pp. 633-640).

RBM code:
Tsogkas S, Kokkinos I, Papandreou G, Vedaldi A. Deep learning for semantic part segmentation with high-level guidance. arXiv preprint arXiv:1505.02438. 2015 May 10.
