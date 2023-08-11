clear;
clc;

dataPath = './demodataset/';   % The dataset path  
trainName = 'trainData';    % The name of the training data  
testName=  'testData';      % The name of the testing data
C = [0.125,0.25,0.5,1,2,4,8];  % Set the range of parameter C
V = [0.125,0.25,0.5,1,2,4,8];  % Set the range of parameter v
% kertype='rbf';
% Delta = [0.001,0.01,0.1,1,10,100,1000];  % RBF kerenel parameter 

kertype='linear';              % Choose the kernel type ('linear' or 'rbf')
Delta=1;   % Delta is a dumb variable for linear kernel

trainName = [dataPath trainName '.mat'];
load(trainName);

testName = [dataPath testName '.mat'];
load(testName);

main(trainData,testData,C,V,kertype,Delta);