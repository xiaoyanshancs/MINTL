function lable = predict(classifier,bag,kertype,delta)
    K = length(classifier);
    result = zeros(1,K);
    for k=1:K
        alphaSign = cell2mat(cellfun(@(x, y) x*y, num2cell(classifier{k}.a'), classifier{k}.s, 'UniformOutput', false));
        temp = abs(alphaSign*kernel(cell2mat(classifier{k}.v),bag,kertype,delta)+classifier{k}.b)-classifier{k}.e;
        result(k) = min(temp);
    end
    [~,idx] = min(result);
    lable = idx;
end

