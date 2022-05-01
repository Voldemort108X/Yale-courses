function dy=hodgkin_huxley_equations(t,y,Iapp,tapp)
% hodgkin_huxley_equations.m
% Contains the Hodgkin-Huxley model for example7_5
% Apply current I app for t < tapp
% Constants
ggK=36; ggNa=120; ggL=0.3;
vK=-12; vNa=115; vL=10.6; 
%Iapp=0;  
Cm=1;
% Equations
v=y(1); n=y(2); m=y(3); h=y(4);
[alpha_n,beta_n,alpha_m,beta_m,alpha_h,beta_h]=rate_constants(v);
if t < tapp
    I=Iapp;
else
    I=0.0;
end
dy=[(-ggK*n^4*(v-vK)-ggNa*m^3*h*(v-vNa)-ggL*(v-vL)+I)/Cm
    alpha_n*(1-n)-beta_n*n
    alpha_m*(1-m)-beta_m*m
    alpha_h*(1-h)-beta_h*h]; 

