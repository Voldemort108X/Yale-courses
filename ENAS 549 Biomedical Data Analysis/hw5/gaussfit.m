% gaussfit.m
% Do Gauss' algorithm for nonlinear fitting
%function beta(1)*t^beta(2)*exp(-beta(3)*t)
% use analytical derivatives
function b=gaussfit(beta,b0,percsd)
 %beta=[1,1,.1];
 %b0=[.5,.5,.05];
 %percsd=10;
 display(beta);
 %create the data
 % skip t=0 since 0^x not defined
 t=(.5:.5:40)';  %column vector
 yper=beta(1)*(t.^beta(2)).*exp(-beta(3)*t);
 n=length(t);
 p=3;
 % simulate noisy data with percsd % error
 y=yper+randn(n,1)*percsd/100.*mean(yper);
 eps=.001;    %fractional change allowed
 b=b0;   % starting guess
 maxchg=1.0;
 maxiter=100;
 iter=0;
 display(b);
 eta1=b(1)*(t.^b(2)).*exp(-b(3)*t);
 while ((maxchg > eps) & (iter < maxiter))
     iter=iter+1;
     % b(n+1)=b(n)+ inv(X'X)X'(y-eta(n))
     % current function evaluation
     eta=b(1)*(t.^b(2)).*exp(-b(3)*t);
     ss=sum((y-eta).^2);
     display(['Iteration ',int2str(iter),' Sum of squares is ',num2str(ss)]);
     % analytical derivatives
     detadb1=(t.^b(2)).*exp(-b(3)*t);
     detadb2=b(1).*log(t).*(t.^b(2)).*exp(-b(3)*t);
     detadb3=b(1)*(t.^b(2)).*(-t).*exp(-b(3)*t);
     xmat=zeros(n,p);
     xmat(:,1)=detadb1;
     xmat(:,2)=detadb2;
     xmat(:,3)=detadb3;
     delta_b=inv(xmat'*xmat)*xmat'*(y-eta);
     maxchg=max(abs(delta_b'./b));
     b=b+delta_b';
     display(b);
 end
 plot(t,yper,'-',t,y,'x',t,eta,'r',t,eta1,'g');
 ylim([0,max(eta)]);
end

