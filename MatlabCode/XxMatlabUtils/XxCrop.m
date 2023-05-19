function img_crop = XxCrop(img, crop_y, crop_x)

[Ny, Nx, ~] = size(img);
midy = round(Ny / 2);
midx = round(Nx / 2);

rx = floor(crop_x / 2);
ry = floor(crop_y / 2);

if mod(crop_y, 2) == 0
    y = midy-ry+1:midy+ry;
else
    y = midy-ry:midy+ry;
end

if mod(crop_x, 2) == 0
    x = midx-rx+1:midx+rx;
else
    x = midx-rx:midx+rx;
end
img_crop = img(y, x, :);

