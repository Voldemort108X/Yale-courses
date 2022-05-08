function F=pet_model(b,t,extra)
% use ode45
% extra hold input function for ode45
nb=length(b);
if nb == 2
    % 1T model
    yzero=0;
else
    yzero=[0.,0.];
end
    
% Set the initial conditions, constants, & time span
extra1=extra;
extra1.k=b;
% without setting Maxstep, ode45 outsmarts the sampling of the input
% function
opt=odeset('MaxStep',1.);
% Integrate the equations
%sol=ode45(@pettis,[0,max(t)],yzero,[],extra1);
sol=ode45(@pettis,[0,max(t)],yzero,opt,extra1);

y=deval(sol,t);
% at this point y is n by 1 or n by 2 
% flag control whether we are fitting the product data only (flag= 1) or
% the product plus free enzyme 
if (nb == 2) 
    F=y;
else
    % return the sum of 2 compartments
    F=y(1,:)+y(2,:);
end
F=F';   %return row vector