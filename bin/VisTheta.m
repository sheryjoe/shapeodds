function VisTheta(Theta, D_x, D_y, withColorBar)

if ~exist('withColorBar' , 'var')
    withColorBar = 0;
end


figure;
subplot(221); VisImageSet2(Theta.w0, D_x, D_y, ['w0'], 1,gca);if withColorBar colorbar(); end
subplot(222); VisImageSet2(Theta.W,  D_x, D_y, ['W'], 1,gca);if withColorBar colorbar(); end
subplot(223); VisImageSet2(ComputeQ(Theta.w0), D_x, D_y, ['q0'], 1,gca);if withColorBar colorbar(); end
subplot(224); VisImageSet2(ComputeQ(Theta.W),  D_x, D_y, ['Q'], 1,gca);if withColorBar colorbar(); end
