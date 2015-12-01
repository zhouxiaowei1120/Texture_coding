% an example on AR database
close all;
clear all;
clc;

dat_ad     =    [cd '\Data\'];
addpath([cd '\utilities\']);
addpath('l1_ls_matlab');
addpath('Data');

% load('H_RData.mat');
% load([dat_ad 'H_TestData.mat']);
TexModel = 'CA_Forest fire';
DataName = ['norml_renderData_' TexModel '.mat'];
load(DataName);

% load([dat_ad 'AR_DR_DAT']);
% you need create this mat by yourself using AR database.
% the mat 'AR_DAT_DAT'contains four dats:
% Dic_Data: the training data in Session 1
% Test_Data:  the testing data in Session 2
% trainlabels:  the training data's labels
% testlabels:   the testing data's labels
% the image size is 60*43, without any preprocessing

trainType = 1;  % 0 representting render; 1 representting height recovery
im_h       = 128;
im_w       = 128;
image_num_str = 1;
image_num_end = 50;
PROC_NUM          =  1550;

if(trainType == 1)
    Dic_Tar      =   trainData(:,1:PROC_NUM);           % RSC�����Ⱦͼ���ֵ�?
    Dic_Data       =   trainTarget(:,1:PROC_NUM);
    Test_Data     =   testTarget(:,image_num_str:image_num_end); % �������Ϊ��Ⱦͼ���ɴ���ɸ߶�ͼ
else
    % RSC��ø߶�ͼ���ֵ�?
    Dic_Data      =   trainData(:,1:PROC_NUM);
    Dic_Tar       =   trainTarget(:,1:PROC_NUM);
    Test_Data     =   testData(:,image_num_str:image_num_end); % �������Ϊ�߶�ͼ���ɴ������Ⱦͼ
end

%Dic_Data = mat2gray(Dic_Data);  % ��һ����0-255
%Dic_Data = zscore(Dic_Data);
%Dic_Tar = mat2gray(Dic_Tar);    % ��һ����0-255
%Dic_Tar = zscore(Dic_Data);
% D_labels          =   trainlabels(1:PROC_NUM);

%Test_Data = zscore(Test_Data);
% testlabels        =   testlabels(1:PROC_NUM-1);
% classids          =   unique(D_labels);
% classnum          =   length(classids);
% clear H_TrainData H_TestData R_TrainData R_TestData;
clear trainData trainTarget;
% PROC_NUM          =   6000;
% Dic_Data = H_TrainData;
% Test_Data= H_TestData;


image_size = im_h*im_w;

nIter             =   2;    % usually only need 2 iterations
residual          =   [];
%eigen_num         =   50;  % the dimensionality of eigenface
lambda            =   0.001;
MEDC              =   [0.8];
BETC              =   [8];

[disc_set,disc_value,Mean_Image]=Eigenface_f(Dic_Data,30);
disc_value       =   sqrt((disc_value));
mean_x           =   255* Mean_Image+0.001*disc_set*disc_value';

% [disc_set,disc_value,Mean_Image]=Eigenface_f(Dic_Data,eigen_num);

 ori_D             =  Dic_Data;  
 ori_D             =  ori_D./ repmat(sqrt(sum(ori_D.*ori_D)),[size(ori_D,1) 1]);  % process the dictionary of height
 
 Tar_D             = Dic_Tar;
 Tar_D             =  Tar_D./ repmat(sqrt(sum(Tar_D.*Tar_D)),[size(Tar_D,1) 1]);  % process the dictionary of height

 median_c   =  MEDC(1);
 beta_c     =  BETC(1);
 ID         =  [];
 code = cell(size(Test_Data,2),1); % save the code of each height image
 squ_sum = 0;
 squ_sum1 = 0;
 squ_sum2 = 0;
 time = cell(size(Test_Data,2),1);
 recon = cell(size(Test_Data,2),1);
 render_ori = cell(size(Test_Data,2),1);
        
for index_pro =1:size(Test_Data,2)
    
        tic;
        residual           =   (Test_Data(:,index_pro)-mean_x).^2;
        residual_sort      =   sort(residual);
        iter               =   residual_sort(ceil(median_c*length(residual))); 
        beta               =   beta_c/iter; 
        w                  =   1./(1+1./exp(-beta*(residual-iter)));
        W                  =   diag(w);
     
        for nit = 1: nIter

        Ori_Train_DAT      =  W* Dic_Data;
        Ori_Test_DAT       =  W* Test_Data(:,index_pro);
        Ori_Train_DAT      =  Ori_Train_DAT./ repmat(sqrt(sum(Ori_Train_DAT.*Ori_Train_DAT)),[size(Ori_Train_DAT,1) 1]);
        Ori_Test_DAT       =  Ori_Test_DAT./ norm(Ori_Test_DAT);
       % [disc_set,disc_value,Mean_Image]=Eigenface_f(Ori_Train_DAT,eigen_num);

%         Train_DAT         =  disc_set'*Ori_Train_DAT;
%         Test_DAT          =  disc_set'*Ori_Test_DAT;
          Train_DAT         =  Ori_Train_DAT;
          Test_DAT          =  Ori_Test_DAT;

        D                 =  Train_DAT;
        D                 =  D./ repmat(sqrt(sum(D.*D)),[size(D,1) 1]);
        y                 =  Test_DAT;
        y                 =  y./norm(y);

%         Glambda = gpuArray(lambda);
%         [Gx,Gstatus]        =  l1_ls(D,y,Glambda);
%         x = gather(Gx);
%         status = gather(Gstatus);
        
        [x,status]        =  l1_ls(D,y,lambda);
        
        ori_y             =  Test_Data(:,index_pro);
        norm_y            =  norm(ori_y,2);
        ori_y             =  ori_y./norm(ori_y,2);


        residual           =   norm_y^2*(ori_y-ori_D*x).^2;
        residual_sort      =   sort(residual);
        iter               =   residual_sort(ceil(median_c*length(residual))); 
        beta               =   beta_c/iter; 
        w                  =   1./(1+1./exp(-beta*(residual-iter)));
        W                  =   diag(w);
        end
        
%         gap1 = [];
%         for class = 1:classnum
%          temp_s  =  x (D_labels == class);
%          z1      =  y-D(:,D_labels == class)*temp_s;
%          gap1(class) = z1(:)'*z1(:);
%         end
%         index = find(gap1==min(gap1));
%         ID(index_pro) = index(1);
recon{index_pro,1} = W*ori_D*x; % reconstruct the input
recon_image=(reshape(recon{index_pro,1},im_h,im_w))';   % convert the vector to image
%recon_image = mat2gray(recon_image);
%imshow(recon_image*255);
if trainType == 0
    namestr = 'sparse_coding_result';
else
    namestr = 'sparse_coding_result_pred_height';
end
filepath = [namestr '/recon/' num2str(index_pro) '.png'];  % write the reconstruct height image
imwrite(recon_image,filepath); % �ع��߶�ͼ����Ⱦͼ

render = W*Tar_D*x; % predict the output
ren_image=(reshape(render,im_h,im_w))';   % convert the vector to image
render_ori{index_pro,1} = mat2gray(render);

filepath = [namestr '/render_ori/' num2str(index_pro) '.png']; % write the predicting render image
imwrite(ren_image,filepath);  % ֱ����0-255��Χ���?

filepath = [namestr '/render255/' num2str(index_pro) '.png']; % write the predicting render image
imwrite(ren_image*255,filepath);  % ֱ�ӳ���255���?

render_image = mat2gray(ren_image);  % ��ͼƬת����0-255֮�䣬��Сֵת��Ϊ0�����ֵ�?55
imshow(render_image);
filepath = [namestr '/render/' num2str(index_pro) '.png']; % write the predicting render image
imwrite(render_image,filepath);


render_image = double(render_image); % ת��Ϊdouble����
if trainType == 1
    ren_samp = testData(:,index_pro); % �����Ⱦͼ��Ӧ�ĸ߶��?
    recon_samp = testTarget(:,index_pro);
else
    ren_samp = testTarget(:,index_pro); % ��ø߶�ͼ��Ӧ����Ⱦ�
    recon_samp = testData(:,index_pro);
end

ren_tar = reshape(ren_samp,im_h,im_w)';
recon_tar = reshape(recon_samp,im_h,im_w)';
squ_sum1 = squ_sum1+sum(sum((recon_image - recon_tar)*(recon_image - recon_tar))); % sum of error between the recovery and original input iamges
squ_sum2 = squ_sum2+sum(sum((render_image - ren_tar)*(render_image - ren_tar))); % sum of error between mat2gray images and the target images
squ_sum = squ_sum+sum(sum((ren_image - ren_tar)*(ren_image - ren_tar)));

code{index_pro,1} = x;  % save the code of each height image
time{index_pro,1} = toc;

fprintf('This is the %d test sample, the running time is %f.\n',index_pro,time{index_pro,1});

end
MSE_1 = squ_sum1/(image_num_end-image_num_str+1)/image_size; % recovery error
MSE_2 = squ_sum2/(image_num_end-image_num_str+1)/image_size; % after mat2gray to compute the MSE
MSE = squ_sum/(image_num_end-image_num_str+1)/image_size;  % before mat2gray to compute the MSE
fprintf('The MSE_1  MSE_2  and  MSE are %f,%f,%f\n',MSE_1,MSE_2,MSE);
filename = ['code/code_' num2str(image_num_end)  datestr(now,30) '.mat'];
save(filename,'code','time','MSE_1','MSE_2','MSE','recon','render_ori');

% reco_rate = sum(ID == testlabels')/length(testlabels);