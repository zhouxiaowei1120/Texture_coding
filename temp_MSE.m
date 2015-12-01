squ_sum = 0;
image_num = 10;
image_size = 32*32;
for i = 1:image_num
    filename =['sparse_coding_result\render\' num2str(i) '.png'];
    ren_pred = imread(filename);
    ren_pred = double(ren_pred)/255;
    filename = ['sparse_coding_result\render_soft\' num2str(i+22600) '.png'];
    ren_soft = imread(filename);
    ren_soft = double(ren_soft(:,:,1))/255;
    a = ren_pred - ren_soft;
    squ_sum = squ_sum+sum(sum((ren_pred - ren_soft)*(ren_pred - ren_soft)));
    fprintf('The %d times squ_sum is %.2f\n',i,squ_sum);
end
MSE_1 = squ_sum/image_num;
MSE_2 = squ_sum/image_num/image_size;
fprintf('The MSE_1 and MSE_2 are %.2f,%.2f\n',MSE_1,MSE_2);