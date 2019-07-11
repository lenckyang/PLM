function [test_accuracy,trainingtime] = PLM(y,b_label,L,parameter1,parameter2,model_number)




  bb_label=b_label;
for count1=1:20
gama=0.01+count1*0.005;

for i=1:1  
    b_label=bb_label(1:600,:);
    test_label=bb_label(601:1000,:);
    yy1=size(bb_label);
    yy1=yy1(2);
    data=b_label(:,1:yy1-1);
b_label=b_label(:,yy1);
    ssdata=bb_label(1:1000,:);
tic
for yxf=1:model_number
    yxf
for wsn=1:L,
    wsn

if wsn==1 
    qqq(wsn)=parameter1;
    qqq1(wsn)=parameter2;
else
     qqq(wsn)=parameter1;
    qqq1(wsn)=1;
end


   
    

      for k = 1:qqq(wsn),
        
        if wsn==1

          rand_sequence=randperm(size(data,1));
    temp_data=data;
    data=temp_data(rand_sequence, :);
    b_label=b_label(rand_sequence);
    termination=size(data,1);
    if termination>=5
    data1=data(1:5,:);
    else
        error('not enough training samples. Please reduce model_number');
    end
    


        else

          data1=[y phi];


          
        end
        
        
        for j = 1:qqq1(wsn),

         if wsn==1
[learn_time77, train_accuracy77,D_Input,BiasofHiddenNeurons1,D_beta,D_YYM,D_beta1,BB,PS,P11,T11,NumberofTestingData]=D_ELM(data1,data,0,1,'sig',4);  %B-ELM

         else
[learn_time77, train_accuracy77,D_Input,BiasofHiddenNeurons1,D_beta,D_YYM,D_beta1,BB,PS,P11,T11,NumberofTestingData]=D_ELM(data1,data,0,1,'sig',4);   %B-ELM

         end
         
         
         
tempH_test=D_Input*P11;
         %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons1(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower('sig')
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
TY1=(H_test' *D_beta)';                       %   TY: the actual output of the testing data
E1=T11 - TY1;
TY=TY1;
for ii=1:4
GXZ1=D_YYM(ii,:)*P11-BB(ii);
switch lower('sig')
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ1'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ1');
end

FYY = mapminmax('reverse',GXZ2',PS(ii));
%FYY=GXZ2;

TY2=FYY'*D_beta1(ii,:);

TestingAccuracy=sqrt(mse(TY2'-E1));

E1=TY2'-E1;
TY=TY+TY2';

end

         
         

         
         
            v = abs(T11-TY);
            v(v>gama)=0;
v(v>0)=1;
num_node(k,j)=sum(v);


if j>=1 ,
    [kkk,iii]=max(num_node(k,:));
    
    if j==iii

        
        best_D_input(:,:,k,wsn)=D_Input;
        best_BiasofHiddenNeurons1(:,:,k,wsn)=BiasofHiddenNeurons1;
        best_D_beta(:,:,k,wsn)=D_beta;
        best_D_YYM(:,:,k,wsn)=D_YYM;
        best_D_beta1(:,:,k,wsn)=D_beta1;
        best_BB(:,:,k,wsn)=BB;
        best_PS(:,:,k,wsn)=PS;
        ineqsat_best(:,:,k,wsn)=v';
        best_label(:,k,wsn)=b_label;
        best_PPP(:,:,k,wsn)=P11;
        best_TY(:,:,k,wsn)=TY;
        best_TTT(:,:,k,wsn)=T11;
        best_num_node(:,k,wsn)=kkk;
     
       
    
    end
    
end

        end

ineqsat=v;

    end


[tl1,tl2]=max(best_num_node(:,:,wsn));
ctj(wsn)=tl1; ctj2(wsn)=tl2;
if wsn>=2
    if ctj(wsn)-ctj(wsn-1)<0
        gama=gama+0;
    end
end
ineqsat_best1=ineqsat_best(:,:,tl2,wsn);
if wsn==1
label=best_label(:,tl2,wsn);
end
PPP=best_PPP(:,:,tl2,wsn);
TTT=best_TTT(:,:,tl2,wsn);
ineqv = max(max(abs(TTT'-best_TY(:,:,tl2,wsn)'),[],2)-gama,0);

ind = find(ineqsat_best1==1);
b_label=label(ind);
[dum,I] = sort(ineqv(ind));
ind = ind(I);
PPP=PPP';
TTT=TTT';
data=[TTT PPP];



phi=PPP(ind,:);
y=TTT(ind);


if wsn==L
    [posi,wsn1]=max(ctj);
    tl2=ctj2(wsn1);
end
clear num_node;
if wsn<L
clear  best_label; clear  best_PPP;
clear  best_TY; clear     best_TTT;
  clear      best_num_node;
  clear      best_phi;


end



if wsn==L
    wsn=wsn1;
    fdafe=1;
    D_Input_best(:,:,yxf)=best_D_input(:,:,tl2,wsn);
    BiasofHiddenNeurons1_best(:,:,yxf)=best_BiasofHiddenNeurons1(:,:,tl2,wsn);
        D_beta_best(:,:,yxf)=best_D_beta(:,:,tl2,wsn);
        D_YYM_best(:,:,yxf)=best_D_YYM(:,:,tl2,wsn);
        D_beta1_best(:,:,yxf)=best_D_beta1(:,:,tl2,wsn);
        BB_best(:,:,yxf)=best_BB(:,:,tl2,wsn);
        PS_best(:,:,yxf)=best_PS(:,:,tl2,wsn);
        ineqsat_best=ineqsat_best(:,:,tl2,wsn);
        ind=find(ineqsat_best==1);
        b_label=label(ind);
        data=[TTT(ind) PPP(ind,:) b_label];
if yxf==1
        data_best1=data;
end
if yxf==2
        data_best2=data;
end
if yxf==3
        data_best3=data;
end
if yxf==4
        data_best4=data;
end
if yxf==5
        data_best5=data;
end
if yxf==6
        data_best6=data;
end
if yxf==7
        data_best7=data;
end
if yxf==8
        data_best8=data;
end
if yxf==9
        data_best9=data;
end
if yxf==10
        data_best10=data;
end


   
        clear b_label; clear ind; clear data; 

        break
end


end

ind=find(ineqsat_best==0);
b_label=label(ind);
data=[TTT(ind) PPP(ind,:)];
panduan=size(data);




if yxf==model_number
    if yxf==10
    
    
    
traindata=[data_best1(:,2:yy1-1) ;data_best2(:,2:yy1-1);data_best3(:,2:yy1-1) ;data_best4(:,2:yy1-1);data_best5(:,2:yy1-1) ;data_best6(:,2:yy1-1);data_best7(:,2:yy1-1) ;data_best8(:,2:yy1-1);data_best9(:,2:yy1-1) ;data_best10(:,2:yy1-1) ];
            traindata1=[data_best1 ;data_best2];
data_best1(:,yy1)=1;
data_best2(:,yy1)=2;
data_best3(:,yy1)=3;
data_best4(:,yy1)=4;
data_best5(:,yy1)=5;
data_best6(:,yy1)=6;
data_best7(:,yy1)=7;
data_best8(:,yy1)=8;
data_best9(:,yy1)=9;
data_best10(:,yy1)=10;
trainlabel=[data_best1(:,yy1);data_best2(:,yy1);data_best3(:,yy1);data_best4(:,yy1);data_best5(:,yy1);data_best6(:,yy1);data_best7(:,yy1);data_best8(:,yy1);data_best9(:,yy1);data_best10(:,yy1)];
    end
    
            if yxf==2
        trainssdata=[data_best1 ;data_best2];
    traindata=[data_best1(:,2:yy1-1) ;data_best2(:,2:yy1-1)];
      traindata1=[data_best1 ;data_best2];
data_best1(:,yy1)=1;
data_best2(:,yy1)=2;
trainlabel=[data_best1(:,yy1);data_best2(:,yy1)];
    end
    
    if yxf==3
    traindata=[data_best1(:,2:yy1-1) ;data_best2(:,2:yy1-1);data_best3(:,2:yy1-1)];
      traindata1=[data_best1 ;data_best2];
data_best1(:,yy1)=1;
data_best2(:,yy1)=2;
data_best3(:,yy1)=3;
trainlabel=[data_best1(:,yy1);data_best2(:,yy1);data_best3(:,yy1)];
    end
    
        if yxf==4
    traindata=[data_best1(:,2:yy1-1) ;data_best2(:,2:yy1-1);data_best3(:,2:yy1-1);data_best4(:,2:yy1-1)];
      traindata1=[data_best1 ;data_best2];
data_best1(:,yy1)=1;
data_best2(:,yy1)=2;
data_best3(:,yy1)=3;
data_best4(:,yy1)=4;
trainlabel=[data_best1(:,yy1);data_best2(:,yy1);data_best3(:,yy1);data_best4(:,yy1)];
        end
    
    break
end


clear  best*;
clear ineqsat_best;
end
end


learningtime(count1)=toc;
[test_accuracy,C11,TT11] = PLM_test_29(test_label,D_Input_best, BiasofHiddenNeurons1_best,D_beta_best,D_YYM_best,D_beta1_best,BB_best,PS_best,traindata,trainlabel,yy1,yxf,traindata1);
 test_accuracy_save(count1,:)=test_accuracy;
clear  best*;
clear ineqsat_best;
end
[u,ui]=min(test_accuracy_save);
test_accuracy=test_accuracy_save(ui(1),1);
trainingtime=learningtime(ui(1));


% Here ineqsat is never empty

%%%%%%%%%%%%%% end of MAXFS %%%%%%%%%%%%%%
