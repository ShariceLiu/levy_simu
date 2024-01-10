
% Simon Godsill, University of Cambridge, 6/10/05

% Kalman filtering recursion - generic updates, based on Godsill and Rayner (1998) (itself based on Harvey (1989)) :

function [a_filt,P_filt,a_pred,P_pred,log_like,y_samp, exp_bit, log_bit]=kalman_update(a,P,y,Z,C_v,T,H,C_e)

M=length(y);

% Prediction step:
for q=1:length(T);
a=T{q}*a;
a_pred{q}=a;

P=T{q}*P*T{q}'+H{q}*C_e{q}*H{q}';


P_pred{q}=P;
end
a=a_pred{1};
P=P_pred{1};

%if (isnan(P))
%    keyboard
%end

% Correction step:

% Kalman gain:
K=P*Z'*inv(Z*P*Z'+C_v);

F=Z*P*Z'+C_v;
mu_y=Z*a;

% Sample from p(y_t|y_{1:t-1}):
y_samp=mu_y+randn(1)*sqrt(F);

w=y-Z*a;    
a_filt=a+K*w;
P_filt=(eye(length(a))-K*Z)*P;    

% Calculate likelihood:

% Exponent:
exp_bit=-w'*inv(F)*w/2;

% The rest:
log_bit=-M/2*log(2*pi)-0.5*log(det(F));

log_like=exp_bit+log_bit;

