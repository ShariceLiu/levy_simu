import numpy as np

def update_stable_langevin(X,drift,theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W, sigma_W, last_t,v1,v2,M1,M2,M3,sigma_v):
    exp_A_delta_t=np.exp(theta*delta_t_i)*eA0+eA1

    X_t_i_det=exp_A_delta_t@X
    drift_t_i_det=exp_A_delta_t@drift

    gamma = 0
    gammas = []
    V_s = []
    # generate gamma_i, V_i, as arrays
    while gamma<c*delta_t_i:
        delta_gamma = np.random.exponential(scale = 1.0)
        gamma = gamma+delta_gamma

        v_i = np.random.uniform(0,delta_t_i)
        while v_i==0:
            v_i = np.random.uniform(0,delta_t_i) # need to make sure v_i is non-zero

        gammas.append(gamma)
        V_s.append(v_i+last_t)

    gammas = np.array(gammas)
    U_i = np.array(V_s)

    Gamma_i_1_alpha=gammas**(-1/alpha)
    Gamma_i_2_alpha=gammas**2

    # TODO: check if these are element-wise product
    sum_0=sum(Gamma_i_1_alpha)
    sum_1=sum(Gamma_i_1_alpha*np.exp(theta*(t_i-U_i)))
    sum_2=sum(Gamma_i_2_alpha*np.exp(2*theta*(t_i-U_i)))
    sum_3=sum(Gamma_i_2_alpha*np.exp(theta*(t_i-U_i)))
    sum_4=sum(Gamma_i_2_alpha)

    m=delta_t_i**(1/alpha)*(sum_0*v2+sum_1*v1)
    S=delta_t_i**(2/alpha)*(sum_2*M1+sum_3*M2+sum_4*M3)

    # centering term
    drift = b_M*(1/theta*(np.exp(theta*delta_t_i)-1)*np.array([1/theta, 1])-delta_t_i*np.array([1/theta, 0]))

    # Linear sde term:
    cov_sde=(np.exp(2*theta*delta_t_i)-1)/(2*theta)*M1+(np.exp(theta*delta_t_i)-1)/(theta)*M2+delta_t_i*M3
    cov_sde=cov_sde*alpha/(2-alpha)*c**(1-2/alpha)

    R=np.linalg.cholesky(sigma_W**2*S+(sigma_W**2)*cov_sde)
    # R_sde=np.linalg.cholesky(cov_sde)

    # generate next x
    X_diff = R.T@np.random.randn(2,1)
    # sde_diff = R_sde.T@np.random.randn(2,1)

    drift_next=mu_W*drift+drift_t_i_det
    X_next=X_t_i_det+mu_W*(m.T[0]-drift)+X_diff.T[0]
    
    y=X_next[1]+sigma_v*np.random.randn(1)

    return X_next,drift_next,y,exp_A_delta_t,m,S,cov_sde,R,drift