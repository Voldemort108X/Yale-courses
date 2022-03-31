x = 1:1:20;
rho = 0:0.1:1;
n = size(x,2);

cnt = 1;
for rho=0:0.1:0.9
%     % construct matrix D
%     for i=1:1:n
%         for j=1:1:n
%             if i==j
%                 D(i,j) = 1;
%             end
%             if i<j
%                 D(i,j) = 0;
%             end
%             if i>j
%                 D(i,j) = rho^(i-j);
%             end
%         end
%     end
%     
%     % construct phi
%     phi = diag(ones(1,n));
%     
%     % calculate psi
%     psi = D*phi*D.';
    
    % construct matrix R
    for i=1:1:n
        for j=1:1:n
            if i==j
                R(i,j) = 1;
            else
                R(i,j) = rho^(abs(i-j));
            end
        end
    end
    % calculate psi
    psi = R;
    
    % calculate covariance estimate
    W = (1-rho^2)*inv(psi);
    X = [ones(1,n); x; log(x); exp(x/10)].';
%     cov_beta = inv(X.'*W*X)*X.'*W*psi*W*X*inv(X.'*W*X)%*(1-rho^2);
    cov_beta = inv(X.'*W*X);
    
    beta1_std(cnt) = sqrt(cov_beta(1,1));
    beta2_std(cnt) = sqrt(cov_beta(2,2));
    beta3_std(cnt) = sqrt(cov_beta(3,3));
    beta4_std(cnt) = sqrt(cov_beta(4,4));
    cnt = cnt+1;
end

plot(0:0.1:0.9, beta1_std);
hold on
plot(0:0.1:0.9, beta2_std);
hold on
plot(0:0.1:0.9, beta3_std);
hold on
plot(0:0.1:0.9, beta4_std);
xlabel('rho');
ylabel('theoretical standard error');
legend('beta1','beta2','beta3','beta4');
