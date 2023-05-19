function output = XxPrctile(img,prct)

img = double(img);
crit = prctile(img(:),prct);
output = mean(img(img > crit));


