 
function [X_next,drift_next,sde_next,y,exp_A_delta_t,m,S,cov_sde,R,drift]=update_stable_langevin(X,drift,sde,theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W, sigma_W, last_t,v1,v2,M1,M2,M3,sigma_v,jump_threshold,B,epsilon,Gammas)

% Deterministic component:
    exp_A_delta_t=exp(theta*delta_t_i)*eA0+eA1;    

    X_t_i_det=exp_A_delta_t*X;
    drift_t_i_det=exp_A_delta_t*drift;
    sde_t_i_det=exp_A_delta_t*sde;
    
   % Generate series of  Gammas:
   % [Y, V, m, sum_Gammas_1_alpha,  delta, Gammas, sum_Gammas_2_alpha, m_resid, V_resid, sum_Gammas_1_alpha_M0, sum_Gammas_2_alpha_M0] = stable_series( c*delta_t_i, alpha, mu_W, sigma_W);

    if (~exist('Gammas')) 
    delta=exprnd(1,max(1,ceil(1.1*c*delta_t_i)),1);
    Gammas=cumsum(delta);
    while (Gammas(end)<c*delta_t_i)
       delta_new=exprnd(1,ceil(0.1*c*delta_t_i),1);
       Gammas=[Gammas; Gammas(end)+cumsum(delta_new)];
    end   
    Gammas=Gammas(Gammas<c*delta_t_i);
    end
    
   % Generate jump times:
    U_i=rand(length(Gammas),1)*delta_t_i+last_t;
    
    Gamma_i=Gammas;
   % semi-heavytailed model: 
    Gamma_i_1_alpha=Gamma_i.^(-1/alpha);%min(Gamma_i.^(-1/alpha).*(1-exp(-B*Gamma_i.^(1/alpha-epsilon))),jump_threshold);
    
    Gamma_i_2_alpha=Gamma_i_1_alpha.^2;
    
    sum_0=sum(Gamma_i_1_alpha);
    sum_1=sum(Gamma_i_1_alpha.*exp(theta*(t_i-U_i)));
    sum_2=sum(Gamma_i_2_alpha.*exp(2*theta*(t_i-U_i)));
    sum_3=sum(Gamma_i_2_alpha.*exp(theta*(t_i-U_i)));
    sum_4=sum(Gamma_i_2_alpha); 

    m=delta_t_i^(1/alpha)*(sum_0*v2+sum_1*v1);
    S=delta_t_i^(2/alpha)*(sum_2*M1+sum_3*M2+sum_4*M3);
    
    % Centering term:
    drift=(alpha>1)*b_M*(1/theta*(exp(theta*delta_t_i)-1)*[1/theta; 1]-delta_t_i*[1/theta; 0]);
   
    % Linear sde term:
    cov_sde=(exp(2*theta*delta_t_i)-1)/(2*theta)*M1+(exp(theta*delta_t_i)-1)/(theta)*M2+delta_t_i*M3;
    cov_sde=cov_sde*alpha/(2-alpha)*c^(1-2/alpha);
    
   % This version not exact as cov_sde should be scaled by sigma_W^2+mu_W^2
    if (min(eigs(sigma_W^2*S+(sigma_W^2)*cov_sde))>0)&&(min(eigs(cov_sde))>0)
        
     %  R=chol(sigma_W^2*S+(sigma_W^2+mu_W^2)*cov_sde);
     R=cholcov(sigma_W^2*S+(sigma_W^2)*cov_sde);
     R_sde=cholcov(cov_sde); 
    else 
       
      % Cheat!!
       disp('ill-conditioned covariance!')
       R=0*S;
       R_sde=0*S;
    end   
   % Generate next x:
   
   % Note this is a different realisation:
    X_diff=R'*randn(2,1);
    
   % Note this is a different realisation:
    sde_diff=R_sde'*randn(2,1);
%     Sigma=inv(inv(cov_sde)+inv(sigma_W^2*S));
%    
%     if (det(Sigma)>1e-12)
%     Sigma_half=chol(Sigma);
%     else
%         disp('ill conditioned Sigma')
%         Sigma_half=0*Sigma;
%     end
%     sde_diff=Sigma*inv(sigma_W^2*S)*X_diff+Sigma_half'*randn(2,1);
    drift_next=mu_W*drift+drift_t_i_det;
    sde_next=sde_t_i_det+sde_diff;
    X_next=X_t_i_det+mu_W*(m-drift)+X_diff;

    
    
    y=X_next(1)+sigma_v*randn(1);
    
    
  