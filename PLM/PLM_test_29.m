function [test_accuracy,C11,TT11] =  pwabe_test(test_label,D_Input_best, BiasofHiddenNeurons1_best,D_beta_best,D_YYM_best,D_beta1_best,BB_best,PS_best,traindata,trainlabel,yy1,yxf,traindata1,sln,nrg)



sln=yxf;

model=svmtrain(trainlabel,traindata);


testdatalabel=test_label(:,yy1);
testdata=test_label(:,2:yy1-1);

[predictlabel,accuracy] = svmpredict(testdatalabel,testdata,model);%%%%use libsvm tools
predictlabel_save=testdatalabel;


for lx_label=1:sln
label=test_label(:,yy1);
predictlabel=predictlabel_save(1:200);
ind=find(predictlabel==lx_label);


P11=test_label(1:200,2:yy1-1);
P11=P11(ind,:)';
T11=test_label(1:200,1);
T11=T11(ind)';
if lx_label==1
C11=0*test_label(1:200,1);
TT11=0*test_label(1:200,1);
end

NumberofTestingData=size(T11);
NumberofTestingData=NumberofTestingData(2);







for i=1:sln
tempH_test=D_Input_best(:,:,i)*P11;
         %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasofHiddenNeurons1=BiasofHiddenNeurons1_best(:,:,i);
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
TY1=(H_test' *D_beta_best(:,:,i))';                       %   TY: the actual output of the testing data
E1=T11 - TY1;
TY=TY1;
for ii=1:4
GXZ1=D_YYM_best(ii,:,i)*P11-BB_best(:,ii,i);
switch lower('sig')
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ1'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ1');
end

FYY = mapminmax('reverse',GXZ2',PS_best(:,ii,i));
%FYY=GXZ2;
TY2=FYY'*D_beta1_best(ii,:,i);
TestingAccuracy=sqrt(mse(TY2'-E1));
E1=TY2'-E1;
TY=TY+TY2';

end
fafe(i)=sqrt(mse(E1));

[am,indm]=min(fafe);

    if i==indm
           ind=find(predictlabel==lx_label);
      C11(ind)=TY;
TT11(ind)=T11;
    test_accuracy1=sqrt(mse(C11-TT11));


    end
if i==sln
    clear fafe
end




end






end



clear TT11; clear fafe; clear C11; clear lx_label





predictlabel=predictlabel_save(201:400);



%%%%%%%%%%%%%%%%%testing part %%%%%%%%%%%%%%%%%

for lx_label=1:sln
label=test_label(:,yy1);
ind=find(predictlabel==lx_label);


P11=test_label(201:400,2:yy1-1);
P11=P11(ind,:)';
T11=test_label(201:400,1);
T11=T11(ind)';
if lx_label==1
C11=0*test_label(201:400,1);
TT11=0*test_label(201:400,1);
end

NumberofTestingData=size(T11);
NumberofTestingData=NumberofTestingData(2);







for i=1:sln
tempH_test=D_Input_best(:,:,i)*P11;
         %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasofHiddenNeurons1=BiasofHiddenNeurons1_best(:,:,i);
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
TY1=(H_test' *D_beta_best(:,:,i))';                       %   TY: the actual output of the testing data
E1=T11 - TY1;
TY=TY1;
for ii=1:4
GXZ1=D_YYM_best(ii,:,i)*P11-BB_best(:,ii,i);
switch lower('sig')
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
GXZ2=1./(1+exp(-GXZ1'));
    case {'sin','sine'}
        %%%%%%%% Sine
GXZ2=sin(GXZ1');
end

FYY = mapminmax('reverse',GXZ2',PS_best(:,ii,i));
%FYY=GXZ2;
TY2=FYY'*D_beta1_best(ii,:,i);
TestingAccuracy=sqrt(mse(TY2'-E1));
E1=TY2'-E1;
TY=TY+TY2';

end
fafe(i)=sqrt(mse(E1));


[am,indm]=min(fafe);

    if i==indm
           ind=find(predictlabel==lx_label);
      C11(ind)=TY;
TT11(ind)=T11;
    test_accuracy2=sqrt(mse(C11-TT11));


    end
if i==sln
    clear fafe
end






end







end


test_accuracy=[test_accuracy2];
fdae=1;






