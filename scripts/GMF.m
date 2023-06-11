%Gaussian matched filter
% Y: Input image

%sigma = 2, L= 9, T= 13, K = 12

function rt = GMF(Y,sigma,L,T,K)

    [M,N] = size(Y);
    x = [-floor(T/2): floor(T/2)];
    tmp1 = exp(-(x.*x)/(2*sigma*sigma)); 
    tmp1 = max(tmp1)-tmp1; 
    ht1 = repmat(tmp1,[L 1]);
    sht1 = sum(ht1(:));
    mean = sht1/(T*L);
    ht1 = ht1 - mean;
    ht1 = ht1/sht1;

    h{1} = zeros(L+6,T+3);
    for i = 1:L
        for j = 1:T
            h{1}(i+3,j+1) = ht1(i,j);
        end
    end

    for k=1:(K-1)
        ag = (180/K)*k;
        h{k+1} = imrotate(h{1},ag,'bicubic','crop');
    end

    for k=1:K
        R{k} = conv2(Y, h{k}, 'same');
    end

    rt = zeros(M,N);
    ER = zeros (1,K);
    for i=1:M
        for j=1:N
            for f=1:K
            ER(f) = R{f}(i,j);  
            end
            rt(i,j) = max(ER);
        end
    end

    rmin = abs(min(rt(:)));
    for m = 1:M
        for n = 1:N
            rt(m,n) = rt(m,n) + rmin;
        end
    end
    rmax = max(max(rt));
    for m = 1:M
        for n = 1:N
            rt(m,n) = round(rt(m,n)*255/rmax);
        end
end
