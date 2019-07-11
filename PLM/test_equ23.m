
%%%%%%%%%%%%%%%%%%%   Dear Reviewer, this is equation (23) dataset %%%%%%%%%%%%%%%%%%%
clear
%%%%% Generating input signal %%%%%%%%%%

u=4*2*(0.5-rand(1,1000));             

%%%%%%%%% initial condition  %%%%%%%%% 
y(1)=0; y(2)=0;                 
y1(1)=0; y1(2)=0;
input=[];
output=[];
x(1,:,:)=[0.5 ,0.7]';

%%%%% Noise level %%%%
%a=0.2;

a=0.5;


%%%%¡¡equ (23)  in the revised manuscript %%%%%%%%%%%%%%%%%%
for t=2:1001
    if 4*y(t-1)-u(t-1)+10<0
       y(t)=-0.4*y(t-1)+u(t-1)+1.5+a*2*(0.5-rand);
                aaa=[y(t-1) u(t-1)];
       input=[input;aaa];
       output=[output ;y(t)];
         y1(t)=1;
    end
    if 4*y(t-1)-u(t-1)+10>=0 && 5*y(t-1)+u(t-1)-6<=0
        y(t)=0.5*y(t-1)-u(t-1)-0.5+a*2*(0.5-rand);
                 aaa=[y(t-1) u(t-1)];
       input=[input;aaa];
       output=[output ;y(t)];
          y1(t)=2;
    end
    if 5*y(t-1)+u(t-1)-6>0
        y(t)=-0.3*y(t-1)+0.5*u(t-1)-1.7+a*2*(0.5-rand);
                 aaa=[y(t-1) u(t-1)];
       input=[input;aaa];
       output=[output ;y(t)];
          y1(t)=3;
    end
    
     
end



y1=y1(2:1001);   


%%%%Normalization%%%%%%%%%%%%
[output1,nrg]=mapminmax(output',0,1);   %%%mapminmax should be run in MATLAB2009a%%%%%%%%%%%%%%%%
output=output1;
input=mapminmax(input');               %%%mapminmax should be run in MATLAB2009a%%%%%%%%%%%%%%%%


data=[output' input' y1'];

label=[output' input' y1'];
%%%%%set parameters%%%%%%%%
L=10;
parameter1=10;
parameter2=10;
model_number=3;
 %[test_accuracy] =PLM_23(data,data,L,parameter1,parameter2,gama);
 [test_accuracy,Trainingtime] =PLM(data,data,L,parameter1,parameter2,model_number);
accuracy=test_accuracy;   %%%can also Consider Table IX (equation (23))




