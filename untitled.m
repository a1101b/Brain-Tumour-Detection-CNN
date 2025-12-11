clc;
clear;

%Get file path for dataset
noDatasetPath = fullfile("C:\Users\adamb\Downloads\archive\Images");

dataset = imageDatastore(noDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

targetSize = [224 224];
% Calculate the number of images for training (70% of the total images)
numTrainFiles = floor(0.7 * numel(dataset.Files));

% Calculate the number of images for each class in the training set
numTrainWithTumor = floor(0.7 * 2185);
numTrainWithoutTumor = numTrainFiles - numTrainWithTumor;

% Split the dataset into training and validation sets
[imdsTrainWithTumor, imdsValidationWithTumor] = splitEachLabel(dataset, numTrainWithTumor, 'randomize');
[imdsTrainWithoutTumor, imdsValidationWithoutTumor] = splitEachLabel(dataset, numTrainWithoutTumor, 'randomize');


% Define preprocessing options using imageDataAugmenter
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ... % Randomly reflect the images horizontally
    'RandYReflection', true, ... % Randomly reflect the images vertically
    'RandRotation', [-10 10], ... % Random rotation between -10 to 10 degrees
    'RandXTranslation', [-20 20], ... % Random horizontal translation between -20 to 20 pixels
    'RandYTranslation', [-20 20]); % Random vertical translation between -20 to 20 pixels

% Concatenate the training sets for both classes
imdsTrain = imageDatastore(cat(1, imdsTrainWithTumor.Files, imdsTrainWithoutTumor.Files), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Create augmentedImageDatastore for training set
augmentedDS_train = augmentedImageDatastore(targetSize, imdsTrain, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');

% Create augmentedImageDatastore for validation sets
augmentedDS_validationWithTumor = augmentedImageDatastore(targetSize, imdsValidationWithTumor, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augmentedDS_validationWithoutTumor = augmentedImageDatastore(targetSize, imdsValidationWithoutTumor, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');

% Combine the validation sets for both classes
imdsValidation = imageDatastore(cat(1, imdsValidationWithTumor.Files, imdsValidationWithoutTumor.Files), ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Create augmentedImageDatastore for validation set
augmentedDS_validation = augmentedImageDatastore(targetSize, imdsValidation, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
%Set input size 
inputSize = [224 224 3];
%Count the number of classes based on the labels
numClasses = length(labelCount.Label);

%Set training options
options = trainingOptions('adam', ...#sgdm
    'InitialLearnRate',0.0003, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedDS_validation, ...
    'ValidationFrequency',25, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'MiniBatchSize',32);

%Set architecture of layers
    layers_5 = [
    imageInputLayer(inputSize)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 512, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    dropoutLayer(0.5) 
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%Analyse and train network
    analyzeNetwork(layers_5)
    net_5 = trainNetwork(augmentedDS_train,layers_5,options);




