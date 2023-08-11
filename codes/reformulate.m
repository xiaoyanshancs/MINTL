function [X,X_sign,P,U,C] = reformulate(MISvm,Sk,notSk,kertype,delta)
%L1  is the number of bags in the subset Sk 
L1 = length(Sk);
%n1 is the number of instances in the subset Sk 
col_nums = cellfun(@(x) size(x, 2), Sk, 'UniformOutput', false);
n1 = sum([col_nums{:}]);
%n2 is the number of instances in the subset notSk 
n2 = size(notSk,2);

%Save the redefined instances (without signs)
X = cell(1,L1+n1+n2);

%Save the signs of the redefined instances
X_sign = cell(1,L1+n1+n2);

%RedefineP
P = zeros(1,L1+n1+n2);

%Redefine U
U = zeros(1,L1+n1+n2);

%RedefineC
C = zeros(1,L1+n1+n2);

index = 1;

alphaSign = cell2mat(cellfun(@(x, y) x*y, num2cell(MISvm.a'), MISvm.s, 'UniformOutput', false));
 

%Obtain the first L1 instances
for i=1:L1
    result = alphaSign*kernel(cell2mat(MISvm.v),Sk{i},kertype,delta)+MISvm.b;
    min_val = min(abs(result)); % Find the minimum value
    min_idx = find(abs(result) == min_val); % Find the index of the minimum value
    X{index} = Sk{i}(:,min_idx);
    temp = result(min_idx);
    nonzero = temp ~= 0;
    signs = ones(size(temp));
    signs(nonzero) = sign(temp(nonzero));
    X_sign{index} = -signs;
    P(index) = 0;
    U(index) = -1;
    C(index) = MISvm.c0;
    index = index+1;
end

%Obtain the L1 to L1+n1 instances
for i=1:L1
    result = alphaSign*kernel(cell2mat(MISvm.v),Sk{i},kertype,delta)+MISvm.b;
    positive_idx = find(abs(result)-0.5-MISvm.e>=0);
    kij = -ones(1,size(result,2));
    kij(positive_idx) = 1;
    positive_idx = find(((abs(result)-0.5-MISvm.e>=0)&result>=0)|(~(abs(result)-0.5-MISvm.e>=0)&~(result>=0)));
    signs = -ones(1,size(Sk{i},2));
    signs(positive_idx) = 1;
    for j=1:size(Sk{i},2)
        X{index} = Sk{i}(:,j);
        X_sign{index} = signs(j);
        P(index) = (kij(j)+1)/2;
        U(index) = kij(j)/2;
        C(index) = MISvm.c2;
        index = index+1;
    end
end

%Obtain the L1+n1 to L1+n instances
result = alphaSign*kernel(cell2mat(MISvm.v),notSk,kertype,delta)+MISvm.b;
nonzero = result ~= 0;
signs = ones(size(result));
signs(nonzero) = sign(result(nonzero));
for i=1:n2
    X{index} = notSk(:,i);
    X_sign{index} = signs(i);
    P(index) = 1;
    U(index) = 1;
    C(index) = MISvm.c1;
    index = index+1;
end

end

