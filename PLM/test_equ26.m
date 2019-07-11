
%%%%%%%%%%%%%%%%%%%   Dear Reviewer, this is equation (26) dataset %%%%%%%%%%%%%%%%%%%
clear
N=1000;
%%%%% Generating input signal %%%%%%%%%%

X= 4*2*(rand(N,3) - 0.5);        % u \in [-4,4]    
%X = 2*(rand(N,3) - 0.5);         % u \in [-1,1]
%%%%%%%%% initial condition  %%%%%%%%% 
y(1)=0; y(2)=0;                 
y1(1)=0; y1(2)=0;
x(1,:,:)=[0.5 ,0.7]';

%%%%% Noise level %%%%
a=2;
%a=0.5;


%%%%¡¡equ (26)  in the revised manuscript %%%%%%%%%%%%%%%%%%




Y = zeros(N,1);
Wtrue = 10*rand(3,3) - 5;
truemode = ceil(2*rand(N,1));




NoiseVariance = 1;
noise = 2*(rand(N,1)-0.5);
for i=1:N
	Y(i) = X(i,:)*Wtrue(:,truemode(i)) +a* noise(i);
end


input=X;
output=Y;

%%%%Normalization%%%%%%%%%%%%
[output]=mapminmax(output',0,1);     %%%mapminmax should be run in MATLAB2009a, the output of cammend mapminmax always changes in different Matlab Versions %%%%%%%%%%%%%%%%
input=mapminmax(input');            %%%mapminmax should be run in MATLAB2009a, the output of cammend mapminmax always changes in different Matlab Versions %%%%%%%%%%%%%%%% 
data=[output' input' truemode];
L=10;
parameter1=30;
parameter2=10;
%model_number=3;
model_number=2;   % this parameter (model_number) can be removed by giving
%Termination conditions
 [test_accuracy,Trainingtime] =PLM(data,data,L,parameter1,parameter2,model_number);





