function mask = XxCalMask(img, ksize, thresh)
    kernel = fspecial('gaussian',[ksize,ksize],ksize);
    fd = imfilter(img,kernel,'replicate');
    kernel = fspecial('gaussian',[100,100],50);
    bg = imfilter(img,kernel,'replicate');
    mask = fd - bg;
%     mask = imfilter(img,kernel,'replicate');
    mask(mask >= thresh) = 1;
    mask(mask ~= 1) = 0;
    mask = logical(mask);
end