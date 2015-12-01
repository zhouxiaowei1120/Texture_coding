function [J] = Random_Block_Occlu(I,r_h,r_w,height,width)
J = I;
% pic = rgb2gray(imread('pic.png'));
pic = imread('pic.png');
J(r_h:r_h+height-1,r_w:r_w+width-1)= imresize(pic,[height width]);
