function [TrainingTime,TrainingAccuracy,D_Input,BiasofHiddenNeurons1,D_beta,D_YYM,D_beta1,BB,PS,P11,T11,NumberofTestingData] = D_ELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction,kkk)

% Usage: elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm-MultiOutputRegression(TrainingData_File, TestingData_File, No_of_Output, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% No_of_Output          - Number of outputs for regression
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression

%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Load training dataset
%train_data=load(TrainingData_File);
%train_data=train_data(:,2:9);

%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

%%%%%%%%%%% Load training dataset
train_data=TrainingData_File;
T=train_data(:,1)';
aaa=T;
P=train_data(:,2:size(train_data,2))';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=TestingData_File;
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;                                    %   Release raw testing data array
T11=TV.T;
P11=TV.P;
NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    
    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;

end                                                 %   end if of Elm_Type
aaa=T;
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
                                        %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = hardlim(tempH);            
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';
     %   Calculate CPU time (seconds) spent for training ELM

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
                                               



TrainingAccuracy=sqrt(mse(T - Y))   ;            %   Calculate training accuracy (RMSE) for regression case
clear H;

D_YYM=[];
D_Input=[];
D_beta=[];
D_beta1=[];
TY=[];
FY=[];
BiasofHiddenNeurons1=[];

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
start_time_train=cputime;

for i=1:kkk
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
BiasofHiddenNeurons1=[BiasofHiddenNeurons1;BiasofHiddenNeurons];
tempH=P'*InputWeight';
YYM=pinv(P')*tempH;
YJX=P'*YYM;

 tempH=tempH';                                         %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;

%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
                                     %   Release the temparary array for calculation of hidden neuron output matrix H

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
OutputWeight=pinv(H') * T';                        % slower implementation
% OutputWeight=inv(H * H') * H * T';                         % faster implementation
Y=(H' * OutputWeight)'; 
%%%%%%%%%%  Updata input weights %%%%%%%
if i==1
    FY=Y;
else
FY=FY+Y;
end
E1=T-Y;
TrainingAccuracy2=sqrt(mse(E1));
Y2=E1'*pinv(OutputWeight);
Y2=Y2';
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        [Y22,PS(i)]=mapminmax(Y2,0.1,0.9);
    case {'sin','sine'}
        %%%%%%%% Sine
       [Y22,PS(i)]=mapminmax(Y2,0,1);
end

Y222=Y2;
Y2=Y22';

T1=(Y2* OutputWeight)';
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
Y3=1./Y2; 
Y3=Y3-1;
Y3=log(Y3);
Y3=-Y3';
    case {'sin','sine'}
        %%%%%%%% Sine
       Y3=asin(Y2)';
end

T2=(Y3'* OutputWeight)';





%Y4=(Y3-BiasMatrix)';

Y4=Y3;
C=2^2;
YYM=(eye(size(P,1))/C+P * P') \ P *Y4';
%YYM=pinv(P')*Y4';
YJX=P'*YYM;

sqrt(mse(YJX-Y4'));

BB1=size(Y4);
BB(i)=sum(YJX-Y4')/BB1(2);
GXZ1=P'*YYM-BB(i);

switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ1'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ1');
end


FYY = mapminmax('reverse',GXZ2,PS(i));

%FYY=GXZ2;
OutputWeight1=pinv(FYY') * E1'; 
FT1=FYY'*OutputWeight1;
FY=FY+FT1';
TrainingAccuracy=sqrt(mse(FT1'-E1));
D_Input=[D_Input;InputWeight];
D_beta=[D_beta;OutputWeight];
D_beta1=[D_beta1;OutputWeight1];
D_YYM=[D_YYM;YYM'];
T=FT1'-E1;


end
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;







    
    %generate input increasment input weight



start_time_train=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%



end_time_train=cputime;
 test_time=end_time_train-start_time_train;
end

%{
fdaf=1;
%删除无用节点
%{
for ii=1:k+1
    if abs(OutputWeight11(ii,1))<0.001 
        OutputWeight11(ii,1)=0;
    end
end
%}



end_time_train=cputime;
TrainingTime=end_time_train-start_time_train   
%TrainingAccuracy=it1;
%%%%%%%%%%% Calculate the output of testing input
tempH_test=InputWeight11*P;
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons11(:,ind);     
tempH_test=tempH_test + BiasMatrix;
 H_test = 1 ./ (1 + exp(-tempH_test));
%TY=(H_test' * OutputWeight11)';   
%train_data=load(TrainingData_File);
%train_data=train_data(:,4:11);
%T=train_data(:,1:No_of_Output)';
%accurcy=sqrt(mse(yy - TY))  



%%%%%%%%%%%%%%%%%%%%%%%%%%%%
start_time_test=cputime;
tempH_test=InputWeight11*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons11(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test' * OutputWeight11)';                       %   TY: the actual output of the testing data
end_time_test=cputime;
TestingTime=end_time_test-start_time_test           %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY))            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2)  
end
%}