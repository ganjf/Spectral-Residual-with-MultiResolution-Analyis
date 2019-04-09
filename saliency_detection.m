clear;
I = im2double(imread('16.jpg'));
%% create lighting feature map
light_f = rgb2gray(I);
% figure;imshow(I_f);
%% create color feature map
R = I(:, :, 1);
G = I(:, :, 2);
B = I(:, :, 3);
[m, n] = size(R);
max_R_G_B = max([max(max(R)),max(max(G)), max(max(B))]);
min_R_G = min([min(min(R)), min(min(G))]);
RG = zeros(m, n, 'double');
BY = zeros(m, n, 'double');
for i  = 1:m
    for j = 1:n
        RG(i,j) = (R(i,j) - G(i,j)) / max_R_G_B;
        BY(i,j) = (B(i,j) - min_R_G) / max_R_G_B;
    end
end
% figure; imshow(RG);
% figure;imshow(BY);
%% create 0,pi/4,pi/2,pi*3/4 direction feature map
d_f_one = gabor_filter(light_f,0);
d_f_two = gabor_filter(light_f,pi/4);
d_f_three = gabor_filter(light_f,pi/2);
d_f_four = gabor_filter(light_f,pi*3/4);

% figure;imshow(d_f_one);
% figure;imshow(d_f_two);
% figure;imshow(d_f_three);
% figure;imshow(d_f_four);
%% create seliency Map of light, color,direction feature with three resolutions
lightSP = {spectralResidual(light_f, 256),...
           spectralResidual(light_f, 128),...
           spectralResidual(light_f, 64)};
colorSP = {spectralResidual(RG, 256),...
           spectralResidual(BY, 256),...
           spectralResidual(RG, 128),...
           spectralResidual(BY, 128),...
           spectralResidual(RG, 64),...
           spectralResidual(BY, 64)};
directionSP = {spectralResidual(d_f_one, 256),...
               spectralResidual(d_f_two, 256),...
               spectralResidual(d_f_three, 256),...
               spectralResidual(d_f_four, 256),...
               spectralResidual(d_f_one, 128),...
               spectralResidual(d_f_two, 128),...
               spectralResidual(d_f_three, 128),...
               spectralResidual(d_f_four, 128),...
               spectralResidual(d_f_one, 64),...
               spectralResidual(d_f_two, 64),...
               spectralResidual(d_f_three, 64),...
               spectralResidual(d_f_four, 64)};
lightAllSP = lightSP;
colorAllSP = {colorSP{1} + colorSP{2},...
              colorSP{3} + colorSP{4},...
              colorSP{5} + colorSP{6}};
directionAllSP = {directionSP{1} + directionSP{1} + directionSP{3} + directionSP{4},...
                directionSP{5} + directionSP{6} + directionSP{7} + directionSP{8},...
                directionSP{9} + directionSP{10} + directionSP{11} + directionSP{12}};
light_saliencyMap = lightAllSP{1} + imresize((lightAllSP{2}+imresize(lightAllSP{3},2,'bilinear')),2,'bilinear');
color_saliencyMap = colorAllSP{1} + imresize((colorAllSP{2}+imresize(colorAllSP{3},2,'bilinear')),2,'bilinear');
direction_saliencyMap = directionAllSP{1} + imresize((directionAllSP{2}+imresize(directionAllSP{3},2,'bilinear')),2,'bilinear');
light_saliencyMap = mat2gray(imfilter(light_saliencyMap, fspecial('gaussian', [10, 20], 2.5)));
color_saliencyMap = mat2gray(imfilter(color_saliencyMap, fspecial('gaussian', [10, 10], 2.5)));
direction_saliencyMap = mat2gray(imfilter(direction_saliencyMap, fspecial('gaussian', [10, 10], 2.5)));

%% kmeans choosing the best feature with best saliencyMap 
% figure;imshow(lightAllSP{3});
figure;imshow(light_saliencyMap);
figure;imshow(color_saliencyMap);
figure;imshow(direction_saliencyMap);
opts = statset('Display','final');
lightKmeansMap = reshape(light_saliencyMap,[size(light_saliencyMap,1)*size(light_saliencyMap,2),1]);
colorKmeansMap = reshape(color_saliencyMap,[size(color_saliencyMap,1)*size(color_saliencyMap,2),1]);
directionKmeansMap = reshape(direction_saliencyMap,[size(direction_saliencyMap,1)*size(direction_saliencyMap,2),1]);

[IDX1,Ctrs1,SumD1,D1] = kmeans(lightKmeansMap,2,'Replicates',10,'Options',opts);
[IDX2,Ctrs2,SumD2,D2] = kmeans(colorKmeansMap,2,'Replicates',10,'Options',opts);
[IDX3,Ctrs3,SumD3,D3] = kmeans(directionKmeansMap,2,'Replicates',10,'Options',opts);
%   [IDX, C, SUMD, D] = KMEANS(X, K) returns distances from each point
%   to every centroid in the N-by-K matrix D.
%   IDX N*1的向量,存储的是每个点的聚类标号
%   Ctrs K*P的矩阵,存储的是K个聚类质心位置
lightDistance = abs(Ctrs1(1) - Ctrs1(2));
colorDistance = abs(Ctrs2(1) - Ctrs2(2));
directionDistance = abs(Ctrs3(1) - Ctrs3(2));

if (lightDistance > colorDistance) && (lightDistance > directionDistance)
    saliencyMap = light_saliencyMap;
    IDX = IDX1;
end
if (colorDistance > directionDistance) && (colorDistance > lightDistance)
    saliencyMap = color_saliencyMap;
    IDX = IDX2;
end
if (directionDistance > colorDistance) && (directionDistance > lightDistance)
    saliencyMap = direction_saliencyMap;
    IDX = IDX3;
end
saliencyMap = im2uint8(saliencyMap);
histgram = imhist(saliencyMap);
% figure;imhist(saliencyMap);
probability = zeros(size(histgram),'double');
[height, width] = size(histgram);
for i = 1:height
    for j = 1:width
        probability(i,j) = histgram(i,j) / sum(histgram);
    end
end
% figure;bar(probability);
histExpect = 0;
histExpect2 = 0;
for i = 1:height
    histExpect = histExpect + probability(i,1) * i;
    histExpect2 = histExpect2 + probability(i,1) * i * i;
end
standardDeviation = sqrt(histExpect2 - (histExpect^2));
output = zeros(size(saliencyMap), 'double');
% for i = 1:size(saliencyMap, 1)
%     for j = 1:size(saliencyMap, 2)
%         if saliencyMap(i,j) > threshold
%             output(i,j) = 1;
%         end
%     end
% end
IDX = reshape(IDX, size(saliencyMap,1), size(saliencyMap,2));
if length(find(IDX == 1)) > length(find(IDX == 2))
    signal = 2;
else
    signal = 1;
end
for i = 1:size(saliencyMap, 1)
    for j = 1:size(saliencyMap, 2)
        if IDX(i,j) == signal
            output(i,j) = 1.0;
        end
    end
end
figure;imshow(I);
figure;imshow(saliencyMap);
figure;imshow(output);
%% just traditional spectral residual
spectralSaliencyMap = spectralResidual(light_f, 64);
[m,n] = size(spectralSaliencyMap);
traditionalOutput = zeros(m,n);
threshold = sum(sum(spectralSaliencyMap)) / (m*n) * 3;
for i = 1:m
    for j = 1:n
        if spectralSaliencyMap(i, j) > threshold
            traditionalOutput(i, j) = 255;
        else
            traditionalOutput(i, j) = 0;
        end
    end
end
figure;imshow(spectralSaliencyMap);
figure;imshow(traditionalOutput);


%% spectral residual
function saliencyMap = spectralResidual(inImg, size)
    inImg = imresize(inImg, [size, size]);
    myFFT = fft2(inImg);
    myLogAmplitude = log(abs(myFFT));
    myPhase = angle(myFFT);
    mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
    saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
    saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [10, 10], 2.5)));
%     saliencyMap = im2uint8(saliencyMap);
end
%% direction feature function
function filtered = gabor_filter(I,theta)
    filter = zeros(11,11,'double');
    x = 0;
    for i = linspace(-8,8,11)
        x = x + 1;
        y = 0;
        for j = linspace(-8,8,11)
            y = y + 1;
            filter(y,x)=compute(i,j,theta);
        end
    end
    filtered = filter2(filter,I,'valid');
    filtered = abs(filtered);
    filtered = filtered/max(filtered(:));
end
    
function gabor_k = compute(x, y, theta)
    delta = 7.0 / 3;
    lamda = 7.0;
    gamma = 1.0;
    x1 = x * cos(theta) + y * sin(theta);
    y1 = (-x) * sin(theta) + y * cos(theta);
    gabor_k = exp(-(x1^2+gamma^2*y1^2)/(2*delta^2)) * cos(2*pi*x1/lamda);
end
% function gabor_k = compute(x,y,f0,theta)
% r = 1; g = 1;
% x1 = x*cos(theta) + y*sin(theta);
% y1 = -x*sin(theta) + y*cos(theta);
% gabor_k = f0^2/(pi*r*g)*exp(-(f0^2*x1^2/r^2+f0^2*y1^2/g^2))*exp(i*2*pi*f0*x1);
% end