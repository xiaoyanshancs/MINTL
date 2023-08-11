function H = calKernel(X,X_sign,kertype,delta)
    n = length(X);
    H = zeros(n,n);
    for i=1:n
       for j=i:n
          H(i,j) = X_sign{i}*kernel(X{i},X{j},kertype,delta)*X_sign{j}'+sum(X_sign{i})*sum(X_sign{j});
          H(j,i) = H(i,j);
       end
    end
end

