run F:\Sketch_cnn/matconvnet-1.0-beta13/matlab/vl_setupnn
load('F:\Sketch_cnn/net_cnn.mat');
load('new_clothing.mat');
net.layers{end}.type = 'softmax';

database = zeros(length(AD), 4096);
for i = 1 : length(AD)
    im = imread(AD(i).address);
    database(i, :) = Get_Feature(im, net, averageImage);
    if mod(i, 10) == 0
        disp(['Finished: ' int2str(i)]);
    end
end
save('database.mat', 'database');