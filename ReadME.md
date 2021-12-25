# ShapeOdds: Variational Bayesian Learning of Generative Shape Models

This repo contains the open-source implementation of our paper: [ShapeOdds: Variational Bayesian Learning of Generative Shape Models. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2231-2242, 2017.](https://openaccess.thecvf.com/content_cvpr_2017/papers/Elhabian_ShapeOdds_Variational_Bayesian_CVPR_2017_paper.pdf)

## Content

- `example.m`: run a demo examples using the Weizmann horses dataset. In this example you can train a shapeodds model, perform inference and robust inference, quantify model generalization, and draw samples from the model.    
- `EstimateShapeOdds.m`: the main function that trains a shapeodds model given a set of silhouttes    
- `bin`: a directory that includes all auxilary functions used by `EstimateShapeOdds.m` and `example.m`    
- `data`: a directory where data used in `example.m` is stored    
- `models`: a directory where you can store learned models and other outputs    


For any questions, please contact Shireen Elhabian (shireen-at-sci-dot-utah-dot-edu)

## Citation

If you use this code in any publication, please cite the following:

**ShapeOdds**

Shireen Y. Elhabian and Ross T. Whitaker. ShapeOdds: Variational Bayesian Learning of Generative Shape Models. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2231-2242, 2017.

**Piecewise Bounds**

Marlin BM, Khan ME, Murphy KP. Piecewise Bounds for Estimating Bernoulli-Logistic Latent Gaussian Models. InICML 2011 Jun 28 (pp. 633-640).

**RBM code**

Tsogkas S, Kokkinos I, Papandreou G, Vedaldi A. Deep learning for semantic part segmentation with high-level guidance. arXiv preprint arXiv:1505.02438. 2015 May 10.
