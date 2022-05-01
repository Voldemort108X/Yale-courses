% example7_5.m - Simulation of the Hodgkin-Huxley model
% using MATLAB function ode45.m to integrate the differential 
% equations that are contained in the file: 
% hodgkin_huxley_equations.m

clc; clear all;
warning off MATLAB:divideByZero

% Evaluate the initial conditions for gating variables
v=0; [alpha_n,beta_n,alpha_m,beta_m,alpha_h,beta_h]=rate_constants(v);
tau_n=1./(alpha_n+beta_n);
n_ss=alpha_n.*tau_n;
tau_m=1./(alpha_m+beta_m);
m_ss=alpha_m.*tau_m;
tau_h=1./(alpha_h+beta_h);
h_ss=alpha_h.*tau_h;
fprintf('\n The following initial conditions of the gating variables are used:') 
fprintf('\n n_ss= %5.4g \n m_ss= %5.4g \n h_ss= %5.4g ', n_ss,m_ss,h_ss)
fprintf('\n They are the resting steady state values of these variables (when v=0).')
disp(' ')
% Integrate the equations
% default for yzero(1) = 8
yzero=[0,n_ss,m_ss,h_ss];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
tspan=[0,20];
Iapp=4;
tapp=2.5;
[t,y]=ode45(@hodgkin_huxley_equations,tspan,yzero,[],Iapp,tapp);
% Evaluate the conductances
ggK=36; ggNa=120;  
gK=ggK*y(:,2).^4; gNa=ggNa*y(:,3).^3.*y(:,4);


    % Plot the results
clf; figure(1); plot(t,y(:,1),'k');
title('Time Profile of Membrane Potential in Nerve Cells')
xlabel('Time (ms)'); ylabel('Potential (mV)')
% % control plot output
% do_plots=1;
% do_plots=0;
% if do_plots 
%     figure(2); plot(t,y(:,2:4));
%     title('Time Profiles of Gating Variables')
%     xlabel('Time (ms)'); ylabel('Gating variables')
%     text(7,0.6,'\leftarrow n(t)'); text(4.5,0.9,'\leftarrow m(t)'); 
%     text(7,0.25,'\leftarrow h(t)')
%     figure(3); plot(t,gK,t,gNa);
%     title('Time Profiles of Conductances')
%     xlabel('Time (ms)'); ylabel('Conductances')
%     text(7,6,'g _K'); text(3.6,25,'g _{Na}'); 
% 
% 
%     % Evaluate the rate constants
%     v=[-100:1:100];
%     [alpha_n,beta_n,alpha_m,beta_m,alpha_h,beta_h]=rate_constants(v);
% 
%     % Evaluating time constants and gating variables at steady state   
%     tau_n=1./(alpha_n+beta_n);
%     n_ss=alpha_n.*tau_n;
%     tau_m=1./(alpha_m+beta_m);
%     m_ss=alpha_m.*tau_m;
%     tau_h=1./(alpha_h+beta_h);
%     h_ss=alpha_h.*tau_h;
% 
%     % Plot the time constants
%     figure(4); plot(v,tau_n,v,tau_m,v,tau_h)
%     axis([-100 100 0 10])
%     title('Time Constants as Functions of Potential')
%     xlabel('Potential (mV)'); ylabel('Time constants (ms)')
%     text(-75,4,'\tau _n'); text(0,0.8,'\tau _m'); text(15,8,'\tau _h');  
% 
%     % Plot the gating variables at steady state 
%     figure(5); plot(v,n_ss,v,m_ss,v,h_ss)
%     axis([-100 100 0 1])
%     title('Gating Variables at Steady State as Functions of Potential')
%     xlabel('Potential (mV)'); ylabel('Gating variables at steady state')
%     text(-35,0.1,'n_\infty'); text(25,0.4,'m_\infty');  text(-20,0.8,'h_\infty'); 
% end     
