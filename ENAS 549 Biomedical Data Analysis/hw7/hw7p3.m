%% Problem 3
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
v_thres = 90; % action potential threshold

cnt_tapp = 1;
for tapp = 0.5:2:20.5
    for Iapp = 1:1:1000
        [t,y]=ode45(@hodgkin_huxley_equations,tspan,yzero,[],Iapp,tapp);
        % Evaluate the conductances
        ggK=36; ggNa=120;  
        gK=ggK*y(:,2).^4; gNa=ggNa*y(:,3).^3.*y(:,4);
        if max(y(:,1))>v_thres
            Iapp_list(cnt_tapp) = Iapp;
            break
        end
    end
    cnt_tapp = cnt_tapp + 1;
end

figure;
plot(0.5:2:20.5, Iapp_list);

% % Plot the results
% clf; figure(1); plot(t,y(:,1),'k');
% title('Time Profile of Membrane Potential in Nerve Cells')
% xlabel('Time (ms)'); ylabel('Potential (mV)')
