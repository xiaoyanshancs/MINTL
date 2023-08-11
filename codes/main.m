function main(trainData,testData,C,V,kertype,Delta)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input parameters:
% trainData: traininig dataset
% testData: testing dataset
% Parameters C and V
% kertype: kernel type ('linear' or 'rbf')
% Delta: rbf kernel paramter. If kertype='linear', Delta is a dumb variable.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
%Get the class labels
class = unique(trainData(:,end));
K = size(class,1);

classifier = cell(1,K);

result = cell(1,size(C,2));
bestRes.acc=0;
bestRes.c0=0;
bestRes.delta=0;
bestRes.v=0;

fprintf('The algorithm starts... \n');

for f=1:size(C,2)
    c0=C(f);
    c1=c0;
    c2=c0;
    acc = zeros(size(Delta,2),size(V,2));
    for g=1:size(Delta,2)
        delta=Delta(g);
        for h=1:size(V,2)
            v=V(h);
            for k=0:K-1
                %Get Sk and notSk
                idx = trainData(:, end) == k; % The last column of trainData is the class label 
                idx_1st_col = trainData(idx, 1);  
                unique_1st_col = unique(idx_1st_col);  
                n_group = length(unique_1st_col);  
                Sk = cell(1,n_group);  
                for i = 1:n_group
                    i_group = find(trainData(:,1)==unique_1st_col(i)); 
                    Sk{i} = trainData(i_group, 2:(size(trainData, 2) - 1))'; 
                end
                notSk = trainData(~idx, 2:(size(trainData, 2) - 1))';

                classifier{k+1} =  Train(Sk,notSk,kertype,delta,c0,c1,c2,v);
            end


            unique_1st_col = unique(testData(:,1)); % The first column of testData is the bag id.
            n_group = length(unique_1st_col);  
            testBag = cell(1,n_group);  
            predict_lable = zeros(1,n_group);
            true_lable = zeros(1,n_group);
            for i = 1:n_group
                    i_group = find(testData(:,1)==unique_1st_col(i));  
                    testBag{i} = testData(i_group, 2:(size(testData, 2) - 1))';  
                    true_lable(i) = testData(i_group(1),end);
                    predict_lable(i) = predict(classifier,testBag{i},kertype,delta);
            end


            predict_lable = predict_lable-1;
            mun = true_lable-predict_lable;
            res = sum(mun==0)/n_group;
            if res>bestRes.acc
                bestRes.acc=res;
                bestRes.c0=c0;
                bestRes.delta=delta;
                bestRes.v=v;
            end
            fprintf('c=%4f  delta=%4f  v=%4f  acc=%4f \n',c0,delta,v,res);
            acc(g,h) = res;
        end
    end
    result{f} = acc;
end

save(['result/result.mat'],'result','bestRes');


