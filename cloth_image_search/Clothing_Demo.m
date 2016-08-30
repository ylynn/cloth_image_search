% Run Clothing_Pretrain First.
f = Get_Feature(im,net,averageImage);

s = sum((database - repmat(f, length(database), 1)) .^ 2, 2);
[~, idx] = sort(s);

figure;
for i=1:21
    subplot(5,5,i)
    I = imread(AD(idx(i)).address);
    imshow(I);
end