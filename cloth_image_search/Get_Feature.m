function [f] = Get_Feature(img,net,averageImage)

TS = 224;
I0 = Get_Contour_Map(img,TS); 
[m,n] = size(I0);
a1 = 1;
a2 = 1;
if m>n
   a2 = max(1,fix((TS - n)/2));
elseif n>m
   a1 = max(1,fix((TS - m)/2));
end
I = zeros(TS,TS);
I(a1:a1+m-1,a2:a2+n-1) = I0;
im = single(255*im2bw(uint8(I),0.2));
im_ = cat(3,im,im,im);
im_ = im_ - averageImage ;
res = vl_simplenn(net, im_);
featVec = res(21).x; 
featVec = featVec(:);
f = featVec'; 
nf = norm(f);
if(nf~=0)
   f = f./nf;
end