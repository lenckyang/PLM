

%%%%%%This problem does not show in our publication, but actualy we guess
%%%%%%that PLM maybe can solve this multi-instance learning problem.


load bank8FM.data
data=bank8FM';
data=mapminmax(data);
data(9,:)=(data(9,:)/2)+0.5;
data=[data(9,:); data(1:8,:)]';
rand_sequence=randperm(size(data,1));
    temp_data=data;
    data=temp_data(rand_sequence, :)';
         output1=data(1,1:4000);
     input1=data(2:9,1:4000);
%Training=data(:,1:300);
%    Testing=data(:,3001:4030);
y1(1:4000)=1;

clear data;
clear rand_sequence;
clear temp_data;


load abalone.dt
data=abalone';
data=mapminmax(data);
data(9,:)=(data(9,:)/2)+0.5;
data=[data(9,:); data(1:8,:)]';
rand_sequence=randperm(size(data,1));
    temp_data=data;
    data=temp_data(rand_sequence, :)';
         output2=data(1,1:4000);
     input2=data(2:9,1:4000);
%Training=data(:,1:300);
%    Testing=data(:,3001:4030);
y1(4001:8000)=2;


clear data
clear rand_sequence;
clear temp_data;



     
     
input=[input1 input2];
output=[output1 output2];


data=[output; input; y1]';

rand_sequence=randperm(size(data,1));
    temp_data=data;
    data=temp_data(rand_sequence, :)';
         output=data(1,:);
     input=data(2:9,:);
     y=output;
     y1=data(10,:);
     
%%%%Normalization%%%%%%%%%%%%
[output]=mapminmax(output,0,1)';     %%%mapminmax should be run in MATLAB2009a, the output of cammend mapminmax always changes in different Matlab Versions %%%%%%%%%%%%%%%%
input=mapminmax(input');            %%%mapminmax should be run in MATLAB2009a, the output of cammend mapminmax always changes in different Matlab Versions %%%%%%%%%%%%%%%% 
data=[output input y1'];
L=10;
parameter1=30;
parameter2=10;
%model_number=3;
model_number=2;   % this parameter (model_number) can be removed by giving
%Termination conditions
 [test_accuracy,Trainingtime] =PLM(data,data,L,parameter1,parameter2,model_number);





