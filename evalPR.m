%% A demo code to compute precision-recall curve for evaluating salient object detection algorithms
% Yao Li, Jan 2014
% please cite our paper "Contextual Hypergraph Modeling for Salient Object
% Detection", ICCV 2013, if you use the code in your research
%% initialization
clear all
close all;clc;
method = 'DSR_sparse'; % name of the salient object method you want to evaluate, you need to change this
dataset = 'ECSSD'; % name of dataset, you need to change this
resultpath = ['./DSR_MSRA-1000map','\*_MSEPG.bmp']; % path to saliency maps, you need to change this
truthpath = 'F:\image datasets\saliency\MSRA\MSRA-1000 binarymasks\*.bmp';%'F:\image datasets\saliency\MSRA\MSRA-1000 binarymasks\*.bmp';%['../../Dataset/',dataset,'_binarymasks/*.bmp']; % path to ground-truth masks, yoiu need to change this
savepath = './MSRA_result/'; % save path of the 256 combinations of precision-recall values
if ~exist(savepath,'dir')
    mkdir(savepath);
end
% resultpath1 = ['./mapnew1/objectness','\*.jpg']; 
dir_im = dir(resultpath);
assert(~isempty(dir_im),'No saliency map found, please check the path!');
dir_tr= dir(truthpath);
assert(~isempty(dir_tr),'No ground-truth image found, please check the path!');
% assert(length(dir_im)==length(dir_tr),'The number of saliency maps and ground-truth images are not equal!')
imNum = length(dir_im);
precision = zeros(256,1);
recall = zeros(256,1);
adaptive_p = zeros(1,imNum);
adaptive_r = zeros(1,imNum);
adaptive_f = zeros(1,imNum);
mae = zeros(1,imNum);
f_beta2 = 0.3;
%% compute pr curve
for n = 1:imNum
  imName = dir_im(n).name;
  input_im = imread([resultpath(1:end-11),imName(1:end-4),resultpath(end-3:end)]);
  truth_im = imread([truthpath(1:end-5),imName(1:end-10),truthpath(end-3:end)]);
  truth_im = truth_im(:,:,1);
  input_im = input_im(:,:,1);
%   if max(max(truth_im))==255
%         truth_im = truth_im./255;
%   end
     if size(input_im,1)~=size(truth_im,1)&&size(input_im,2)~=size(truth_im,2)
       input = zeros(size(truth_im));
       h = (size(truth_im,1)-size(input_im,1))/2;
       w = (size(truth_im,2)-size(input_im,2))/2;
       input(h+1:size(input_im,1)+h,w+1:size(input_im,2)+w) = input_im;
       input_im = input; 
   end
   for threshold = 0:255
    index1 = (input_im>=threshold);
    truePositive = length(find(index1 & truth_im));
    groundTruth = length(find(truth_im));
    detected = length(find(index1));
    if truePositive~=0
     precision(threshold+1) = precision(threshold+1)+truePositive/detected;
     recall(threshold+1) = recall(threshold+1)+truePositive/groundTruth;
    end
    end
    display(num2str(n));
    adaptive_T = 2.0*sum(input_im(:))/(size(input_im,1)*size(input_im,2));
    if sum(sum(input_im >adaptive_T))==0
        adaptive_T = adaptive_T/1.2;
    end
    segment_im = zeros(size(input_im));
    segment_im(input_im>adaptive_T) = 255;
%     imwrite(segment_im,[savepath imName(1:end-4) '.jpg'],'jpg');
    adaptive_tp =0;
    for i =1:size(input_im,1)
        for j = 1:size(input_im,2)
            if input_im(i,j) >adaptive_T&&truth_im(i,j)>0.5
                adaptive_tp = adaptive_tp +1;
            end
        end
    end
%     tp = sum(sum(smap>adaptive_T&&gt>0.5));
    p = adaptive_tp/(sum(sum(input_im>adaptive_T))+1e-10);
    r = adaptive_tp/(sum(sum(truth_im>0.5))+1e-10);
    adaptive_p(n) =  p;
    adaptive_r(n) =  r;
    adaptive_f(n) = (1+f_beta2)*p*r/(f_beta2*p + r + 1e-10);
    mae(n) = sum(sum(abs(double(input_im)/255 -double(truth_im))))/(size(truth_im,1)*size(truth_im,2));
end
precision = precision./imNum;
recall = recall./imNum;
pr = [precision'; recall'];
fid = fopen([savepath dataset, '_', method, '_PRCurve.txt'],'at');
fprintf(fid,'%f %f\n',pr);
fclose(fid);
disp('Done!');
adaptive_p = sum(adaptive_p)/imNum;
adaptive_r = sum(adaptive_r)/imNum;
adaptive_f = sum(adaptive_f)/imNum;
adaptive = [ adaptive_p adaptive_r adaptive_f];
fid = fopen([savepath dataset, '_', method, '_apPRCurve.txt'],'at');
fprintf(fid,'%f %f\n',adaptive);
fclose(fid);
disp('Done!');
mae = sum(mae)/imNum;
fid = fopen([savepath dataset, '_', method, '_mea.txt'],'at');
fprintf(fid,'%f %f\n',mae);
fclose(fid);
disp('Done!');

figure(1);
hold on;
set(gca,'FontSize',14);
axis([0 1 0 1]);
grid on;
plot(recall, precision, '-', 'color', [0 1 0], 'linewidth', 2);
