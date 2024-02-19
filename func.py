import numpy as np
import scipy


def returnQ(delta_t, l, k_mu1, k_mu2):
    theta = -l
    M1 = np.matrix([[1/theta**2, 1/theta],[1/theta, 1]])
    M2 = np.matrix([[-2/theta**2, -1/theta],[-1/theta, 0]])
    M3 = np.matrix([[1/theta**2,0],[0,0]])
    Q_delta_t1 = (np.exp(2*theta*delta_t)-1)/2/theta*M1+ (np.exp(theta*delta_t)-1)/theta*M2 + delta_t*M3

    Q_delta_t = np.matrix(np.zeros((4,4)))
    Q_delta_t[:2,:2] = Q_delta_t1 * (1+delta_t*k_mu1**2)
    Q_delta_t[2:,2:] = Q_delta_t1 * (1+delta_t*k_mu2**2)

    return Q_delta_t

def construct_Cmu(vs, sigma_mu, T):
    # construct variance of mu, 1d, vs must be ordered
    # add final t
    C = np.matrix(np.zeros((len(vs)+1, len(vs)+1)))
    for i in range(len(vs)):
        C[i:,i] += vs[i]*sigma_mu**2
        C[i,i:] += vs[i]*sigma_mu**2
        C[i,i] -= vs[i]*sigma_mu**2
    C[-1,-1] += T*sigma_mu**2

    return C

def dim2todim1(mu_p, P_p):
    # import pdb; pdb.set_trace()
    mu_p1 = np.matrix([mu_p[0].item(), mu_p[1].item(), mu_p[4].item()]).T
    mu_p2 = np.matrix([mu_p[2].item(), mu_p[3].item(), mu_p[5].item()]).T
    P_p1 = np.matrix(np.zeros((3,3)))
    P_p1[:2,:2] = P_p[:2,:2]
    P_p1[:2,2] = P_p[:2, 4]
    P_p1[2,:2] = P_p[4,:2]
    P_p1[2,2] = P_p[4,4]

    P_p2 = np.matrix(np.zeros((3,3)))
    P_p2[:2,:2] = P_p[2:4,2:4]
    P_p2[:2,2] = P_p[2:4, 5]
    P_p2[2,:2] = P_p[5,2:4]
    P_p2[2,2] = P_p[5,5]

    return [mu_p1, mu_p2], [P_p1, P_p2]

def dim1todim2(mu_ps, P_ps):
    mu_p = np.matrix(np.zeros(6)).T
    mu_p[:2] = mu_ps[0][:2]
    mu_p[2:4] = mu_ps[1][:2]
    mu_p[4] = mu_ps[0][2]
    mu_p[5] = mu_ps[1][2]

    P_p = np.matrix(np.zeros((6,6)))
    P_p[:2,:2] = P_ps[0][:2,:2]
    P_p[:2, 4] = P_ps[0][:2,2]
    P_p[4,:2] = P_ps[0][2,:2]
    P_p[4,4] = P_ps[0][2,2]

    P_p[2:4,2:4] = P_ps[1][:2,:2]
    P_p[2:4, 5] = P_ps[1][:2,2]
    P_p[5,2:4] = P_ps[1][2,:2]
    P_p[5,5] = P_ps[1][2,2]

    return mu_p, P_p

def constructA2Ce(vs, gammas, T, l, alpha, c):
    # jump together, changing mu
    A2 = np.matrix(np.zeros((3,len(vs)+3)))
    m12 = -(np.exp(-T*l)-1)/l
    m22 = np.exp(-T*l)
    eAt = np.matrix([[1,m12],[0,m22]])

    A2[:2,:2] = eAt
    s_c_dt = 0.0
    for i in range(len(vs)):
        v_i = vs[i]
        gamma = gammas[i]
        # calculate mean delta t^c      
        f1 = -np.exp(-l*(T-v_i))/l+1/l
        f2 = np.exp(-(T-v_i)*l)
        f_delta_t_v_i_m = np.matrix([f1,f2]).T
        f_delta_t_v_i = np.matrix([[f1,0],[f2,0]])
        m_c_y_c = T**(1/alpha)*gamma**(-1/alpha)*f_delta_t_v_i_m
        s_c_dt += T**(2/alpha)*gamma**(-2/alpha)*np.matmul(f_delta_t_v_i,f_delta_t_v_i.T)
        A2[:2,2+i] = m_c_y_c

    A2[-1,-1] = 1
    # Q_delta_t = returnQ(T)
    C_e = s_c_dt # + alpha/(2-alpha)*c**(1-2/alpha)*Q_delta_t # C_e

    return A2, C_e
