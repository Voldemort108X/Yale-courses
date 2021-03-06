%function importfile(fileToRead1)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%  Auto-generated by MATLAB  using the Open file Wizard
% (click on generate code)

% Import the file
newData1 = importdata('rec_occ.csv');

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end

% extract variables
free=data(:,1);    % plasma level
bnd=data(:,2);    % binding amount
x=free;
y=bnd;
figure(1);

plot(x,y,'o');
xlabel('Free');
ylabel('Bound');

% do nonlinear fit
%function is recBinding
%  parameter list is [Bmax,Kd]
% initial guess
% let Bmax be max value of binding
% let Kd be average x value
b0=[max(y),mean(x)];
p=2;
n=length(x);
options=optimset('display','off','Algorithm','levenberg-marquardt');
[b,ss,residual,exitflag,output]=lsqcurvefit(@recbinding,b0,x,y,[],[],options);
yfit=residual+y;
% predicted covbariance matrix is k(X'WX)^-1 where k = s/(n-p)
k=ss/(n-p);
% create X - partial exponetial partial
xmat=zeros(n,p);
delta=.001;
beta=b;
for j=1:p
    beta1=beta;
    beta1(j)=(1+delta)*beta(j);
    y1=recbinding(beta1,x);
    xmat(:,j)=(y1-yfit)./(delta*beta(j));
end
bcov=inv(xmat'*xmat)*k;
bpredsd=(diag(bcov).^0.5)';
% increase sig digits for output
%format('long');
display('Nonlinear fit for Bmax and Kd');
display(b);
display(bpredsd);
% calculate smooth curve
xx=0:max(x);
yy=recbinding(b,xx);
plot(x,y,'o',xx,yy);
xlabel('Free');
ylabel('Bound');
title('Nonlinear fit');
% re-plot on semi log plot
figure(2);
xx=0:(2*max(x));
yy=recbinding(b,xx);
semilogx(x,y,'o',xx,yy);
xlabel('Free');
ylabel('Bound');
title('Nonlinear fit');
% Plot residuals with zero-line
figure(3);
xx=[min(x),max(x)];yy=[0,0];
plot(x,y-yfit,'o',xx,yy,'--');title('Residuals from nonlinear fit');

% now redo as Scatchard linear fit
figure(4);
x=bnd;
y=bnd./free;
% B/F = (Bmax/Kd) -(1/Kd)*B
plot(x,y,'o');
% do linear fit
xmat=zeros(n,p);
xmat(:,1)=1;
xmat(:,2)=x;
b2=inv(xmat'*xmat)*xmat'*y;
yfit=xmat*b2;
s=sum((y-yfit).^2)/(n-p);
bcov=inv(xmat'*xmat)*s;
b2predsd=(diag(bcov).^0.5)';
% increase sig digits for output
%format('long');
display('Scatchard linear fit');
display(b2);
display(b2predsd);
xx=0:max(x);
yy=b2(1)+b2(2)*xx;
plot(x,y,'o',xx,yy);
xlabel('Bound');
ylabel('Bound/Free');
title('Scatchard fit');
% b2 is Bmax/Kd and -1/Kd
% so Kd is -1/b2(2)
% Bmax=b2(1)*kd
kd=-1/b2(2);
bmax=kd*b2(1);
b=[bmax,kd];
display(b);
