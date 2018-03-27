%% Image Segmentation using VGG16 architecture 

%% initialization
clear all;
close all;

%% Extract CT Image to PNG
sourceFiles = dir(fullfile(pwd,'Source','CTImages','*.nii')); 

for i = 1:size(sourceFiles,1)
    nii_no = i;
    im_address = strcat(sourceFiles(nii_no).folder,'\',sourceFiles(nii_no).name);
    
    [filepath,name] = fileparts(im_address);
    
    if ~exist(fullfile(filepath,name),'dir')
        mkdir(filepath,name)
    end
    
    im_info = niftiinfo(im_address);
    im = niftiread(im_info);
    % calibrated image
    image = im2single(im);
    
    % save new image in RGB format (VGG requirement)
    for j=1:size(im,3)
       fileName = strcat(filepath,'\',name,'\',name,'-',num2str(j),'.png');
       tempImage = imresize(image(:,:,j),0.5);
       imnew = cat(3,tempImage,tempImage,tempImage); 
       imwrite(imnew,fileName); 
    end
    
end

%% Extract CT Labels to grayscale PNG (1 dimension)
sourceFiles = dir(fullfile(pwd,'Source','CTLabels','*.nii')); 

for i = 1:size(sourceFiles,1)
    nii_no = i;
    im_address = strcat(sourceFiles(nii_no).folder,'\',sourceFiles(nii_no).name);
    
    [filepath,name] = fileparts(im_address);
    
    if ~exist(fullfile(filepath,name),'dir')
        mkdir(filepath,name)
    end
    
    im_info = niftiinfo(im_address);
    im = niftiread(im_info);
    image = uint8(im*50); % multiply with 50 to enhance image contrast
    
    for j=1:size(im,3)
       fileName = strcat(filepath,'\',name,'\',name,'-',num2str(j),'.png');
       imwrite(imresize(image(:,:,j),0.5),fileName); 
    end
    
end

%% Build Image Database
imSourceDir = fullfile(pwd,'Source','CTImages');
imSourceds = imageDatastore(imSourceDir,'IncludeSubfolder',true);


%% Build Label Database
classes = [
     "Background"
     "Liver"
     "Tumor"
     ];
 
imLabelIDs = {
           [0;]   % background
           [50;]  % liver
           [100;] % lesion
          };
          
imLabelDir = fullfile(pwd,'Source','CTLabels');
imLabelds  = pixelLabelDatastore(imLabelDir,classes,imLabelIDs,'IncludeSubFolder',true);


%% Check Image and Label Overlay
I = readimage(imSourceds,154); % part of image with liver and tumor
C = readimage(imLabelds,154);
categories(C)
B = labeloverlay(I,C);
figure
imshow(B)

%% Visualize the pixel count for the Label
tbl = countEachLabel(imLabelds)

frequency = tbl.PixelCount/sum(tbl.PixelCount);
figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes))
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')

%% Prepare Training and Test Sets with ratio 60:40
[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionImageData(imSourceds,imLabelds);
numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

%% Create Network
imageSize = [256 256 3]; % image dimension
numClasses = numel(classes);
lgraph = segnetLayers(imageSize,numClasses,'vgg16'); % specify using vgg16

%% Balance Classes by giving class Weights
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

%% Update Segnet layer 
lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');

%% Select training option
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency', 2);

%% Data Augmentation
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);

%% Start Training
datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);

[net, info] = trainNetwork(datasource,lgraph,options);

%% Test Network on one image
I = read(imdsTest);
C = semanticseg(I, net);
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);

% Compared Result
expectedResult = read(pxdsTest);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

% Statistic Result
iou = jaccard(C, expectedResult);
table(classes,iou)

%% Validated Network Model
pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
metrics.DataSetMetrics


function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionImageData(imds,pxds)

% Set initial random state for example reproducibility.
rng(0);
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
N = round(0.60 * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end

