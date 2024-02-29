import numpy as np
import matplotlib.pyplot as plt
import scipy

T = 1
c = 20
alpha = 1
beta = 0.5
mu = 0.0
sigma = 1.0
l = 0.001
k_v = 10
alpha_w = 0.000000000001
beta_w = 0.000000000001
mu = [0.1,0.05]


def gamma_process_jumps(T = 1, alpha = 1, beta = 0.5, c = 20):
    J_is = []
    V_is = []
    gamma = 0
    # generate gammas
    while gamma<c*T:
        delta_gamma = np.random.exponential(scale = 1.0)
        gamma = gamma+delta_gamma

        # using gamma to represent jumps by inverse-levy measure method
        J_i = 1/(alpha/beta*(np.exp(beta/alpha**2*gamma)-1))
        # acceptance probability
        a_i = (1+alpha/beta*J_i)*np.exp(-alpha/beta*J_i)
        # accept with prob=a_i
        W_i = np.random.uniform(size=1)

        if W_i<=a_i:
            # reject
            J_is.append(J_i)
            v_i = np.random.uniform(0,T)
            while v_i==0:
                v_i = np.random.uniform(0,T) # need to make sure v_i is non-zero
            V_is.append(v_i)

    # sort jump time
    J_is = [x for _,x in sorted(zip(V_is, J_is))]
    V_is.sort()

    return J_is, V_is

def VG_jumps(J_is, V_is, mu = 0.0, sigma = 1.0):
    """
    take gamma jumps and return variance gamma jumps
    """
    vg_J_is = []
    for j_i in J_is:
        vg_J = np.random.normal(loc = mu*j_i, scale= sigma**2*j_i)
        vg_J_is.append(vg_J)
    
    return vg_J_is, V_is

def VGSSM(T = 1.0, l = 0.001):
    J_is, V_is = gamma_process_jumps(T = T)
    m_c_dt = 0
    s_c_dt = 0
    for j_i, v_i in zip(J_is, V_is):
        f1 = -np.exp(-l*(T-v_i))/l+1/l
        f2 = np.exp(-(T-v_i)*l)
        f_delta_t_v_i_m = np.matrix([f1,f2,f1,f2]).T
        f_delta_t_v_i = np.matrix([[f1,0],[f2,0],[0,f1],[0,f2]])
        m_c_dt += j_i*f_delta_t_v_i_m
        s_c_dt += j_i*np.matmul(f_delta_t_v_i,f_delta_t_v_i.T)

    return m_c_dt, s_c_dt

def single_ite_particle_filtering_para(y_n, log_weight_p, mu_p, P_p, E_ns, n, T=1.0, resample = False, k_v = 10):
    # mu_p = [mu1, mu2, .. muP], P_p = [P1, P2, .. PP]
    num_particles = len(mu_p) # list of list of lists, because length if each particle is not fixed, P*n*I
    if resample:
        indices = np.random.choice(list(range(num_particles)), size = num_particles, p = np.exp(log_weight_p))
        mu_p = [mu_p[i] for i in indices]
        P_p = [P_p[i] for i in indices]
        
#         weight_p = np.ones(num_particles)*(1/num_particles)
        log_weight_p = np.log(np.ones(num_particles)*(1/num_particles))
    
    C = np.matrix(np.zeros((2,6)))
    C[0,0] = 1
    C[1,2] = 1 # observation matrix
    R = np.matrix(np.zeros)
    R = np.identity(2)*k_v**2

    m12 = -(np.exp(-T*l)-1)/l
    m22 = np.exp(-T*l)
    e_Adt = np.matrix([[1,m12,0,0],[0,m22,0,0],[0,0,1,m12],[0,0,0,m22]])
    # construct A
    A = np.matrix(np.zeros((6,6)))
    A[:4,:4] = e_Adt
    A[4:,4:] = np.identity(2)
    B = np.matrix(np.zeros((6,4)))
    B[:4,:4] = np.identity(4)
    
    mu_p_nextn = []
    P_p_nextn = []
    log_weights_p_nextn = np.ones(num_particles)
    for p in range(num_particles):
#         print(p)
        m_c_dt, s_c_dt = VGSSM(T=T)

        A[0,4] = m_c_dt[0]
        A[1,4] = m_c_dt[1]
        A[2,5] = m_c_dt[2]
        A[3,5] = m_c_dt[3]

        mu_n_prev_n = A*mu_p[p]
        P_n_prev_n = A*P_p[p]*A.T + B* s_c_dt*B.T
        y_hat_n_prev_n = C*mu_n_prev_n
        sigma_n_prev_n = C*P_n_prev_n*C.T + R
        K = P_n_prev_n*C.T*np.linalg.inv(sigma_n_prev_n)
        mu_n_n = mu_n_prev_n+K*(y_n-y_hat_n_prev_n)
        P_n_n = (np.identity(6) - K*C)*P_n_prev_n

        mu_p_nextn.append(mu_n_n)
        P_p_nextn.append(P_n_n)

        # count sigma_w in
        E_ns[p,n] = -(y_n-y_hat_n_prev_n).T*np.linalg.inv(sigma_n_prev_n)*(y_n-y_hat_n_prev_n)/2
        beta_w_post_p = beta_w - sum(E_ns[p,:])
        log_like = -0.5*np.log(np.linalg.det(sigma_n_prev_n))-(alpha_w+n/2)*np.log(beta_w_post_p)\
                    +(alpha_w+n/2-1/2)*np.log(beta_w - sum(E_ns[p,:(n-1)]))+\
                    scipy.special.loggamma(n/2+alpha_w)-scipy.special.loggamma(n/2+alpha_w-1/2) # -2/2*np.log(2*np.pi)
    
        # normalise weights
        log_w_p_nextn = log_weight_p[p]+log_like
        log_weights_p_nextn[p] = log_w_p_nextn 
    
    # normalise weights
    try:
        log_weights_p_nextn = log_weights_p_nextn-np.log(sum(np.exp(log_weights_p_nextn)))
    except:
        import pdb;pdb.set_trace()
    
    return log_weights_p_nextn, mu_p_nextn, P_p_nextn, E_ns

def simulate_VGSSM(N=500, T = 1.0, sigma = 1.0, mu = [0.1,0.05], k_v = 10):
    x = np.matrix([0,0,0,0]).T # vertical matrix
    x_ns = np.zeros((N,4))
    t_ns = np.zeros(N)

    m12 = -(np.exp(-T*l)-1)/l
    m22 = np.exp(-T*l)
    e_Adt = np.matrix([[1,m12,0,0],[0,m22,0,0],[0,0,1,m12],[0,0,0,m22]])

    for n in range(N):
        m_c_dt, s_c_dt = VGSSM(T=T)
        m_c_dt[:2]*=mu[0]
        m_c_dt[2:]*=mu[1]
        m_delta_t = np.matmul(e_Adt,x)+m_c_dt
        s_delta_t = s_c_dt*sigma**2

        x = np.random.multivariate_normal(np.array(m_delta_t.T)[0], s_delta_t.T) # horizontal array

        x_ns[n,:]=x
        x = np.matrix(x).T # transform x to be a vertical matrix
        t_ns[n]=n*T

    noise_sig = sigma*k_v
    noise = np.random.normal(0, noise_sig, (N, 4))
    noise[:,1] /= 20
    noise[:,3] /= 20
    y_ns = x_ns+noise # noisy data
    y_ns = y_ns[:,(0,2)]

    return x_ns, t_ns, y_ns

def particle_filter(y_ns, N = 200, num_particles = 10, T = 1):
    n_mus = []
    n_Ps = []
    n_ws = []
    # num_particles = 100

    # initialize x
    mu_p_1 = np.matrix(np.zeros(6)).T
    # mu_p_1[4:,0] = np.matrix([[mu_w1],[mu_w1]])
    mu_p = [mu_p_1]*num_particles
    P_p = [np.matrix(np.identity(6))*100]*num_particles
    log_weight_p = np.log(np.ones(num_particles)*(1/num_particles))

    E_ns = np.zeros((num_particles, N+1)) # store exp likelihood of y

    for n in range(N):
        y_n = np.matrix(y_ns[n]).T
        if n%5 == 0:
            resample = True
        else:
            resample = False
        log_weight_p, mu_p, P_p, E_ns = single_ite_particle_filtering_para(y_n, log_weight_p, mu_p, P_p, E_ns, n+1, T = T, resample = resample)
        
        n_mus.append(mu_p)
        n_Ps.append(P_p)
        n_ws.append(log_weight_p)

    n_mus = np.array(n_mus)
    n_Ps = np.array(n_Ps)
    n_Ps *= sigma**2 # if marginalizing sigma
    n_ws = np.array(n_ws)

    return n_mus, n_Ps, n_ws

def examine_results(n_mus, n_Ps, n_ws, tgt_mu = [0.1,0.05]):
    N = len(n_mus)

    average = np.zeros((N, 6))
    std3 = np.zeros((N, 6))
    avg_P = np.zeros((N, 6, 6))

    for i in range(N):
        for d in range(6):
            average[i,d]=np.dot(np.array(n_mus[i,:,d,:].T)[0], np.exp(n_ws[i,:]))
            for j in range(6):
                avg_P[i,d,j] = np.dot(np.array(n_Ps[i,:,d,j].T), np.exp(n_ws[i,:]))+\
                np.dot((np.array(n_mus[i,:,d,:].T)[0]-average[i,d])*(np.array(n_mus[i,:,j,:].T)[0]-average[i,j]), np.exp(n_ws[i,:]))
                
            std3[i,d]=np.sqrt(avg_P[i,d,d])*3

    # plot error region
    dim = 4
    plt.figure()
    plt.subplot(2,1,1)
    plt.fill_between(range(N),average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                    color='gray', alpha=0.2)
    plt.ylim([-0.5,0.5])
    tgt = np.ones(N)*tgt_mu[0]
    plt.plot(tgt)
    plt.plot(average[:,dim])
    plt.legend(['pred','tgt'])
    plt.title('X axis')

    dim = 5
    plt.subplot(2,1,2)
    plt.fill_between(range(N),average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                    color='gray', alpha=0.2)
    plt.ylim([-0.5,0.5])
    tgt = np.ones(N)*tgt_mu[1]
    plt.plot(tgt)
    plt.plot(average[:,dim])
    plt.legend(['pred','tgt'])
    plt.title('Y axis')
    plt.show()
    plt.savefig('VG/mus.png')

def main():
    x_ns, t_ns, y_ns = simulate_VGSSM()
    plt.figure()
    plt.plot(x_ns[:,0],x_ns[:,2])
    plt.plot(y_ns[:,0],y_ns[:,1])
    plt.savefig('VG/data.png')

    n_mus, n_Ps, n_ws = particle_filter(y_ns, num_particles=100)
    examine_results(n_mus, n_Ps, n_ws)


if __name__ == "__main__":
    main()