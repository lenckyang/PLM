
%%%%%%%%%%%%%%%%%%%   Dear Reviewer, this is equation (24) dataset %%%%%%%%%%%%%%%%%%%
clear
N = 1000;
%%%%% Generating input signal %%%%%%%%%%

X= 4*2*(rand(N,2) - 0.5);          

%%%%%%%%% initial condition  %%%%%%%%% 
y(1)=0; y(2)=0;                 
y1(1)=0; y1(2)=0;
x(1,:,:)=[0.5 ,0.7]';

%%%%% Noise level %%%%
a=1;
%a=0.2;


%%%%¡¡equ (24)  in the revised manuscript %%%%%%%%%%%%%%%%%%



Y = zeros(N,1);
Wtrue =[-0.9 0.7;1 -1] ;

truemode = (mod(1:N,60)>30)+1;

noise = 2*(rand(N,1)-0.5);
for i=1:1000
	Y(i) = X(i,:)*Wtrue(:,truemode(i)) + a* noise(i);
end


input=X;
output=Y;

%%%%Normalization%%%%%%%%%%%%
[output]=mapminmax(output',0,1);     %%%mapminmax should be run in MATLAB2009a, the output of cammend mapminmax always changes in different Matlab Versions %%%%%%%%%%%%%%%%
input=mapminmax(input');            %%%mapminmax should be run in MATLAB2009a, the output of cammend mapminmax always changes in different Matlab Versions %%%%%%%%%%%%%%%% 
data=[output' input' truemode'];

L=10;
parameter1=30;
parameter2=10;
model_number=3;
%model_number=2;   % this parameter (model_number) can be removed by giving
%Termination conditions
 [test_accuracy,Trainingtime] =PLM(data,data,L,parameter1,parameter2,model_number);

 






