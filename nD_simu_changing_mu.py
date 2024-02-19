import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy

from func import construct_Cmu, dim1todim2, dim2todim1, constructA2Ce, returnQ

# changing mu/ jump sep generating data
l = 0.005
c = 20
N = 200
T = 1
sigma_w = 0.033
sigma_mu1 = 0.03
sigma_mu2 = 0.015
alpha = 0.9

k_mu1 = sigma_mu1/sigma_w
k_mu2 = sigma_mu2/sigma_w


def simu_langevin_changing_mu():
    x = np.matrix([0,0,0,0]).T # vertical matrix
    x_ns = np.zeros((N,4))
    t_ns = np.zeros(N)
    u_ns = np.zeros((N,2))

    u1 = 0.1 # start of u
    u2 = 0.05

    vs_nt = []
    gammas_nt = []

    for n in range(N):
        gamma = 0
       
        vs = []
        gammas = []
        while gamma<c*T:
            delta_gamma = np.random.exponential(scale = 1.0)
            gamma = gamma+delta_gamma

            v_i = np.random.uniform(0,T)          
            vs.append(v_i)
            gammas.append(gamma)
       
        # sort v, gamma with increasing v
        gammas = [x for _, x in sorted(zip(vs, gammas), key=lambda pair: pair[0])]
        vs.sort()
        gammas_nt.append(gammas)
        vs_nt.append(vs)
        v0 = 0.0
           
        for v_i, gamma in zip(vs, gammas):
            dt_i = v_i - v0
            # generate u, as a brownian motion
            u1_i = np.random.normal(u1, sigma_mu1*np.sqrt(dt_i))
            u2_i = np.random.normal(u2, sigma_mu2*np.sqrt(dt_i))

            # calculate mean delta t^c      
            f1 = -np.exp(-l*(v_i-v_i))/l+1/l
            f2 = np.exp(-(v_i-v_i)*l)
            f_delta_t_v_i_m = np.matrix([f1,f2,f1,f2]).T
            f_delta_t_v_i = np.matrix([[f1,0],[f2,0],[0,f1],[0,f2]])
           
            # mean
            m_c_dt_i = T**(1/alpha)*gamma**(-1/alpha)*f_delta_t_v_i_m
            m_c_dt_i[0:2] *= u1_i
            m_c_dt_i[2:] *= u2_i
           
            # variance
            s_c_dt_i = T**(2/alpha)*gamma**(-2/alpha)*np.matmul(f_delta_t_v_i,f_delta_t_v_i.T)

            # centering term
            c1 = -(1-np.exp(-l*dt_i))/l**2 + dt_i/l
            c2 = (1-np.exp(-l*dt_i))/l
            Z_c_bar_i = alpha/(alpha-1)*c**(1-1/alpha)* np.matrix([u1_i*c1,u1_i*c2,u2_i*c1,u2_i*c2]).T

            # Q_delta_t = returnQ(dt_i, l, k_mu1, k_mu2)*sigma_w**2
            S_delta_t = sigma_w**2*s_c_dt_i #+ alpha/(2-alpha)*c**(1-2/alpha)*Q_delta_t

            m12 = -(np.exp(-dt_i*l)-1)/l
            m22 = np.exp(-dt_i*l)
            m_delta_t = np.matmul(np.matrix([[1,m12,0,0],[0,m22,0,0],[0,0,1,m12],[0,0,0,m22]]),x)+m_c_dt_i
            if alpha > 1:
                m_delta_t -= Z_c_bar_i

            x = np.random.multivariate_normal(np.array(m_delta_t.T)[0], S_delta_t.T) # horizontal array
            x = np.matrix(x).T # transform x to be a vertical matrix
           
            # updata variables
            u1 = u1_i
            u2 = u2_i
            v0 = v_i

        # generate last u_n
        u1 = np.random.normal(u1, sigma_mu1*np.sqrt(T-v0))
        u2 = np.random.normal(u2, sigma_mu2*np.sqrt(T-v0))
        u_ns[n,0] = u1
        u_ns[n,1] = u2

        # centering term
        dt_i = T - v0
        c1 = -(1-np.exp(-l*dt_i))/l**2 + dt_i/l
        c2 = (1-np.exp(-l*dt_i))/l
        Z_c_bar_i = alpha/(alpha-1)*c**(1-1/alpha)* np.matrix([u1_i*c1,u1_i*c2,u2_i*c1,u2_i*c2]).T

        Q_delta_t = returnQ(dt_i, l, k_mu1, k_mu2)*sigma_w**2
        S_delta_t = alpha/(2-alpha)*c**(1-2/alpha)*Q_delta_t # last dt_i, no jump

        m12 = -(np.exp(-dt_i*l)-1)/l
        m22 = np.exp(-dt_i*l)
        m_delta_t = np.matmul(np.matrix([[1,m12,0,0],[0,m22,0,0],[0,0,1,m12],[0,0,0,m22]]),x)
        if alpha > 1:
            m_delta_t -= Z_c_bar_i

        x = np.random.multivariate_normal(np.array(m_delta_t.T)[0], S_delta_t.T) # horizontal array
        x_ns[n,:]=x
        x = np.matrix(x).T # transform x to be a vertical matrix
        t_ns[n]=n*T
       
    return x_ns, t_ns, u_ns, gammas_nt, vs_nt

if __name__ == '__main__':

    x_ns, t_ns, u_ns, gammas_nt, vs_nt = simu_langevin_changing_mu()

    plt.figure()
    plt.plot(u_ns[:,0], u_ns[:,1])
    plt.title('Mean of noise')

    # add noise
    k_v = 100 # 80
    noise_sig = sigma_w*k_v

    noise = np.random.normal(0, noise_sig, (N, 4))
    y_ns = x_ns+noise # noisy data

    plt.figure()
    plt.plot(y_ns[:,0], y_ns[:,2])
    plt.plot(x_ns[:,0],x_ns[:,2])
    plt.legend(['noisy', 'true'])
    y_ns = y_ns[:,(0,2)]

# prepare algorithm for later use
alpha_w = 0.000000000001
beta_w = 0.000000000001


sigma_mus = [sigma_mu1, sigma_mu2]
def sampling_latent_variables_para(T, alpha, mu_p, P_p, gammas, vs):
    gammas = []
    vs = []
    
    gamma = 0
    while gamma<c*T:
        delta_gamma = np.random.exponential(scale = 1.0)
        gamma = gamma+delta_gamma
        
        v_i = np.random.uniform(0,T)
        while v_i==0:
            v_i = np.random.uniform(0,T) # need to make sure v_i is non-zero

        gammas.append(gamma)
        vs.append(v_i)

    # order jumps
    gammas = [x for _,x in sorted(zip(vs, gammas))]
    vs.sort()

    A1 = np.zeros((3+len(vs),3))
    A1[:2,:2] = np.identity(2)
    A1[2:, 2] += 1
    B1 = np.zeros((3+len(vs), 1+len(vs)))
    B1[2:,:] = np.identity(1+len(vs))
    B2 = np.zeros((3,2))
    B2[:2,:2]=np.identity(2)

    mu_ps, P_ps = dim2todim1(mu_p, P_p)

    for d in range(2):
        # iterate through all dimensions
        Ce1 = construct_Cmu(vs, sigma_mus[d], T)
        mu_p1d = mu_ps[d]
        P_p1d = P_ps[d]

        mu_p1d = A1*mu_p1d
        P_p1d = A1*P_p1d*A1.T + B1* Ce1*B1.T

        A2, Ce2 = constructA2Ce(vs, gammas, T, l, alpha,c)
        Ce2 *= sigma_w**2 # assume known sigma_w
        mu_p1d = A2*mu_p1d
        P_p1d = A2*P_p1d*A2.T + B2* Ce2*B2.T

        mu_ps[d]=mu_p1d
        P_ps[d] = P_p1d
    
    mu_p, P_p = dim1todim2(mu_ps, P_ps)
    # print(mu_p, P_p)
    
    return mu_p, P_p

def single_ite_particle_filtering_bytimestep(y_n, log_weight_p, mu_p, P_p, E_ns, gammas, vs, n, resample = False):
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
    R = np.identity(2)*noise_sig**2
    
    mu_p_nextn = []
    P_p_nextn = []
    log_weights_p_nextn = np.ones(num_particles)
    for p in range(num_particles):
        mu_n_prev_n, P_n_prev_n = sampling_latent_variables_para(1, alpha, mu_p[p], P_p[p], gammas, vs)

        y_hat_n_prev_n = C*mu_n_prev_n
        sigma_n_prev_n = C*P_n_prev_n*C.T + R
        K = P_n_prev_n*C.T*np.linalg.inv(sigma_n_prev_n)
        mu_n_n = mu_n_prev_n+K*(y_n-y_hat_n_prev_n)
        P_n_n = (np.identity(6) - K*C)*P_n_prev_n

        mu_p_nextn.append(mu_n_n)
        P_p_nextn.append(P_n_n)

#         w_p_nextn = weight_p[p]*stats.multivariate_normal(np.array(mu_n_n.T)[0], P_n_n).pdf(np.array(y_n.T)[0])
#         weights_p_nextn[p] = w_p_nextn
        # count sigma_w in
        norm_sigma_n_prev_n = sigma_n_prev_n/sigma_w**2
        E_ns[p,n] = -(y_n-y_hat_n_prev_n).T*np.linalg.inv(norm_sigma_n_prev_n)*(y_n-y_hat_n_prev_n)/2
        beta_w_post_p = beta_w - sum(E_ns[p,:])
        log_like = -0.5*np.log(np.linalg.det(sigma_n_prev_n))-(alpha_w+n/2)*np.log(beta_w_post_p)\
                    +(alpha_w+n/2-1/2)*np.log(beta_w - sum(E_ns[p,:(n-1)]))+\
                    scipy.special.loggamma(n/2+alpha_w)-scipy.special.loggamma(n/2+alpha_w-1/2) # -2/2*np.log(2*np.pi)
    
        # normalise weights
        # log_like = -(y_n-y_hat_n_prev_n).T*np.linalg.inv(sigma_n_prev_n)*(y_n-y_hat_n_prev_n)/2-0.5*np.log(np.linalg.det(sigma_n_prev_n))-2/2*np.log(2*np.pi)
        log_w_p_nextn = log_weight_p[p]+log_like
#         log_w_p_nextn = log_weight_p[p]+np.log(stats.multivariate_normal(np.array(y_hat_n_prev_n.T)[0], sigma_n_prev_n).pdf(np.array(y_n.T)[0]))
        log_weights_p_nextn[p] = log_w_p_nextn 
    
    # normalise weights
    try:
        log_weights_p_nextn = log_weights_p_nextn-np.log(sum(np.exp(log_weights_p_nextn)))
    except:
        import pdb;pdb.set_trace()
#     weights_p_nextn = weights_p_nextn/sum(weights_p_nextn)
    
    return log_weights_p_nextn, mu_p_nextn, P_p_nextn, E_ns

if __name__ == '__main__':
    n_mus = []
    n_Ps = []
    n_ws = []
    num_particles = 1

    # initialize x
    mu_p_1 = np.matrix(np.zeros(6)).T
    # mu_p_1[4:,0] = np.matrix([[mu_w1],[mu_w1]])
    mu_p = [mu_p_1]*num_particles
    P_p = [np.matrix(np.identity(6))*10]*num_particles
    log_weight_p = np.log(np.ones(num_particles)*(1/num_particles))

    E_ns = np.zeros((num_particles, N+1)) # store exp likelihood of y

    for n in range(N):

        y_n = np.matrix(y_ns[n]).T

        gammas = gammas_nt[n]
        vs = vs_nt[n]

        if n%5 == 0:
            resample = True
        else:
            resample = False
        log_weight_p, mu_p, P_p, E_ns = single_ite_particle_filtering_bytimestep(y_n, log_weight_p, mu_p, P_p, E_ns, gammas, vs, n+1, resample = resample)
        
        n_mus.append(mu_p)
        n_Ps.append(P_p)
        n_ws.append(log_weight_p)

    n_mus = np.array(n_mus)
    n_Ps = np.array(n_Ps)
    n_ws = np.array(n_ws)

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
    plt.ylim([max(average[25:,dim] + std3[25:,dim]),min(average[25:,dim] - std3[25:,dim])])
    plt.plot(average[:,dim])
    plt.plot(u_ns[:,0])
    plt.fill_between(range(N),average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                    color='gray', alpha=0.2)
    plt.legend(['pred','tgt'])
    # plt.title('X axis')

    dim = 5
    plt.subplot(2,1,2)
    plt.ylim([max(average[25:,dim] + std3[25:,dim]),min(average[25:,dim] - std3[25:,dim])])
    plt.plot(average[:,dim])
    plt.plot(u_ns[:,1])
    plt.fill_between(range(N),average[:,dim] - std3[:,dim], average[:,dim] + std3[:,dim],
                    color='gray', alpha=0.2)
    plt.legend(['pred','tgt'])
    # plt.title('Y axis')
    plt.show()
    # plt.savefig('predictmu/changing_mu_a09_timestep.png')

    # plt.legend(['mu 1','mu 2'])
    # plt.savefig('Langevin_nonzero_mus_fixjump.png')
    print(average[-1,4],average[-1,5])