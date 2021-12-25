function grid = getGridPatches(grid)
% GETGRIDPATCHES Construct a grid of square patches on an image.

imsize   = grid.imageSize;
overlap	 = grid.overlap;
gridSize = grid.size;
patchSize = (imsize + overlap.*(gridSize - 1)) ./ gridSize;
assert(all(isintegern(patchSize)),'Invalid combination of grid size and overlap');
assert(patchSize(1) == patchSize(2), 'Non-square patches')

% We only consider square patches for now!!
patchSize = patchSize(1);
nPatches  = prod(gridSize);
patches   = false([imsize, nPatches]);
patchSquare = [1:patchSize; 1:patchSize]; 
for j=1:gridSize(2)
    for i=1:gridSize(1)
        patchIndex = (j-1)*gridSize(1) + i;
        patches(patchSquare(1,:), patchSquare(2,:), patchIndex) = true;
        % shift patch vertically
        patchSquare(1,:) = patchSquare(1,:) + patchSize - overlap; 
        assert(nnz(patches(:,:,patchIndex))==patchSize^2)
    end
    % shift patch horizontally
    patchSquare = [1:patchSize; patchSquare(2,:) + patchSize - overlap];
end
grid.patches = patches;
grid.patchSize = patchSize;
















% function grid = getGridPatches(imsize, grid)
% % GETGRIDPATCHES Construct a grid of square patches on an image.
% % 
% % patches = false([opts.imageSize, opts.grid.nPatches]);
% % patchSquare = [1:opts.grid.patchSize; 1:opts.grid.patchSize];
% % for j=1:opts.grid.size(2)
% %     for i=1:opts.grid.size(1)
% %         patchIndex = (j-1)*opts.grid.size(1) + i;
% %         opts.grid.patches(patchSquare(1,:), patchSquare(2,:), patchIndex) = true;
% %         patchSquare(1,:) = patchSquare(1,:) + opts.grid.patchSize;
% %         assert(nnz(opts.grid.patches(:,:,patchIndex))==opts.grid.patchSize^2)
% %     end
% %     % Shift patch square to new coordinates
% %     patchSquare(1,:) = 1:opts.grid.patchSize;
% %     patchSquare(2,:) = patchSquare(2,:) + opts.grid.patchSize;
% % end
% % patches = reshape(patches, [] , opts.grid.nPatches);
% 
% overlap		= grid.overlap;
% gridSize  = grid.size;
% 	
% patchSize = (imsize + overlap.*(gridSize - 1)) ./ gridSize;
% assert(all(isintegern(patchSize)),'Invalid combination of grid size and overlap');
% assert(patchSize(1) == patchSize(2), 'Non-square patches')
% 
% % We only consider square patches for now
% patchSize = patchSize(1);
% nPatches  = prod(gridSize);
% patches   = false([imsize, nPatches]);
% patchSquare = [1:patchSize; 1:patchSize]; 
% for j=1:gridSize(2)
%     for i=1:gridSize(1)
%         patchIndex = (j-1)*gridSize(1) + i;
%         patches(patchSquare(1,:), patchSquare(2,:), patchIndex) = true;
%         % shift patch vertically
%         patchSquare(1,:) = patchSquare(1,:) + patchSize - overlap; 
%         assert(nnz(patches(:,:,patchIndex))==patchSize^2)
%     end
%     % shift patch horizontally
%     patchSquare = [1:patchSize; patchSquare(2,:) + patchSize - overlap];
% end
% grid.patches = patches;
% grid.patchSize = patchSize;
% 
% epiPatchSize =  patchSize - ((overlap-1)/2)+1;
% grid.epiPatchSize = epiPatchSize;
% cnt_epi = 0;
% if grid.use_epitome,
% 	for kv=1:(overlap-1)/2
% 		for kh=1:(overlap-1)/2
% 			cnt_epi = cnt_epi + 1;
% 			zs = zeros(patchSize,patchSize);
% 			zs((kv-1) + [1:epiPatchSize],(kh-1) + [1:epiPatchSize]) = 1;
% 			epitome(:,:,cnt_epi) = zs;
% 			epitome_idxs(:,cnt_epi) = find(zs(:));
% 		end
% 	end
% 	grid.epitome			= epitome;
% 	grid.epitome_idxs = epitome_idxs;
% 	grid.nTransl			= size(epitome,3);
% 	
% 	patchSize			= epiPatchSize;
% 	overlap				= 2*epiPatchSize - imsize(1);
% 	patches_epi   = false([imsize, nPatches]);
% 	patchSquare		= [1:patchSize; 1:patchSize];
% 	for j=1:gridSize(2)
% 		for i=1:gridSize(1)
% 			patchIndex = (j-1)*gridSize(1) + i;
% 			patches_epi(patchSquare(1,:), patchSquare(2,:), patchIndex) = true;
% 			% shift patch vertically
% 			patchSquare(1,:) = patchSquare(1,:) + patchSize - overlap;
% 			assert(nnz(patches_epi(:,:,patchIndex))==patchSize^2)
% 		end
% 		% shift patch horizontally
% 		patchSquare = [1:patchSize; patchSquare(2,:) + patchSize - overlap];
% 	end
% 	grid.patches_epi	= patches_epi;
% end