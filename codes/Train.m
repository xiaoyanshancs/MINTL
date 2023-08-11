function res =  Train(Sk,notSk,kertype,delta,c0,c1,c2,v)
%Sk is a cell, containing all the bags in class k
%notSk is a matrix, containing all the instances not belonging to class k. The instances are column vectors.

    Jt = 0.01;
    J = 0.01;

    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    x0 = [];
    options = optimset;     
    options.LargeScale = 'off';
    options.Display = 'off';

    %L1 is the number of bags in the subset Sk 
    L1 = length(Sk);
    %n1 is the number of instances in the subset Sk 
    col_nums = cellfun(@(x) size(x, 2), Sk, 'UniformOutput', false);
    n1 = sum([col_nums{:}]);
    %n2 is the number of instances in the subset notSk 
    n2 = size(notSk,2);
    
    n = L1+n1+n2;
    
    MISvm.c0 = c0;
    MISvm.c1 = c1;
    MISvm.c2 = c2;
    %Initialize alpha
    MISvm.a = rand(n, 1);
    %Initialize the positive instances; for each bag in the subset Sk, the first instance is selected as the initial positive instance
    v1 = cell(1,L1);
    for i=1:L1
        v1{i} = Sk{i}(:,1);
    end
    Sk_mat = cell2mat(Sk);
    v2 = mat2cell(Sk_mat, size(Sk_mat, 1), ones(1, size(Sk_mat, 2)));
    v3 = mat2cell(notSk, size(notSk, 1), ones(1, size(notSk, 2)));
    MISvm.v = cat(2,v1,v2,v3);
    
    %Initialize sign
    s_mat = sign(randn(1, n));
    MISvm.s = num2cell(s_mat);
    %Initialize b
    alphaSign = cell2mat(cellfun(@(x, y) x*y, num2cell(MISvm.a'), MISvm.s, 'UniformOutput', false));
    MISvm.b = sum(alphaSign);
    
    MISvm.e = 0.1;
    iteration = 1;
    while true
    
        [X,X_sign,P,U,C] = reformulate(MISvm,Sk,notSk,kertype,delta);

        A = [-eye(n);eye(n);-U];
        b = [zeros(n,1);C';v];
        f = -P';
        H = calKernel(X,X_sign,kertype,delta);
        [x,fval,~,~] = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);

        MISvm.a = x;
        MISvm.v = X;
        MISvm.s = X_sign;
        alphaSign = cell2mat(cellfun(@(x, y) x*y, num2cell(MISvm.a'), MISvm.s, 'UniformOutput', false));
        MISvm.b = sum(alphaSign);
        idx = find(x>0.01);
        %Calculate ek
        ek = 0;
        for i=1:size(idx,1)
            result = alphaSign*kernel(cell2mat(MISvm.v),MISvm.v{idx(i)},kertype,delta)*MISvm.s{idx(i)}'+MISvm.b*sum(MISvm.s{idx(i)});
            ek = ek + (result-P(idx(i)))/U(idx(i));
        end
        ek = ek/size(idx,1);
        MISvm.e = ek;

        J = fval;
        %Stopping criterion
        if abs((Jt-J)/J) < 0.1 | iteration>20
            break
        end
        Jt = J;
        iteration = iteration+1;
    end
    res = MISvm;
    
end

