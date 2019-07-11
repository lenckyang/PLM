
%%%%%%%%%%%%%%%%%%%   Dear Reviewer, this is equation (29) dataset %%%%%%%%%%%%%%%%%%%
clear
%%%%% Generating input signal %%%%%%%%%%

u=2*(0.5-rand(1,1000));          %%%%%%  u \in [-1,1] %%%%%%     

%%%%%%%%% initial condition  %%%%%%%%% 
y(1)=0; y(2)=0;                 
y1(1)=0; y1(2)=0;
input=[];
output=[];
x(1,:,:)=[0.5,-0.5]';

%%%%% select noise level  %%%
%a=10;
a=2;



%%%%¡¡equ (29)  in the revised manuscript %%%%%%%%%%%%%%%%%%
for t=2:1001
    if [1 2]*x(t-1,:,:)'+0.3<0
wkn=x(t-1,:,:);
ww1=wkn(1);
ww2=wkn(2);
        x(t,:,:)=[0 sin(ww1) ; 0 0.4]*x(t-1,:,:)';
        y(t)=[1 0]*10*sin(x(t,:,:))';
        y1(t)=1;
 aaa=[x(t-1,:,:)];
    input=[input;aaa];
    output=[output ;y(t)];
    else
        wkn=x(t-1,:,:);
ww1=wkn(1);
ww2=wkn(2);
        x(t,:,:)=[0 sin(ww1); 0.4*(1/2) 0.4-(1/2)]*x(t-1,:,:)'+[0 -0.1]'+[0.1*a*(rand-1) 0.1*a*(rand-1)]';  %%%a=10; a=1; a=2
        y(t)=[0.4 -1]*10*cos(x(t,:,:))';
        y1(t)=2;
         aaa=[x(t-1,:,:) ];
    input=[input;aaa];
    output=[output ;y(t)];
    end
end



y1=y1(2:1001);   


%%%%Normalization%%%%%%%%%%%%
[output1,nrg]=mapminmax(output',0,1);   %%%mapminmax should be run in MATLAB2009a%%%%%%%%%%%%%%%%
output=output1;
input=mapminmax(input');               %%%mapminmax should be run in MATLAB2009a%%%%%%%%%%%%%%%%


data=[output' input' y1'];

label=[output' input' y1'];
L=10;
parameter1=30;
parameter2=10;
model_number=2;
 %[test_accuracy] =PLM_23(data,data,L,parameter1,parameter2,gama);
 [test_accuracy,Trainingtime] =PLM(data,data,L,parameter1,parameter2,model_number);
accuracy=test_accuracy; 




