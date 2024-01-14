import numpy as np
import matplotlib.pyplot as plt
from tools import update_stable_langevin

M=1000
alpha=1.4
mu_W=0.1
sigma_W=0.033
T=500

# Dimension of observation vector:
M_v=1

# Observation noise scaling parameter:
kappa_V=10

sigma_v=kappa_V*sigma_W

# Initial prior scaling parameter for mu_W:
kappa_W=100

# Prior for sigma_W^2 (IG):
alpha_W=0.000000000001
beta_W=0.000000000001

# Langevin model:

# Mean reversion coefficient:
theta=-0.005
A = np.array([[0,1],[0,theta]])

eA0 = np.array([[0,1/theta],[0,1]])
eA1 = np.array([[1,-1/theta],[0,0]])

M1 = np.array([[1/theta**2, 1/theta],[1/theta, 1]])
M2 = np.array([[-2/theta**2, -1/theta],[-1/theta, 0]])
M3 = np.array([[1/theta**2, 0],[0,0]])

v1 = np.array([[1/theta],[1]])
v2 = np.array([[-1/theta],[0]])

time_axis = np.linspace(0, T, num=T+1)

# initial value
X_true = np.zeros([2,T+1])
drift_true = np.zeros([2,T+1])
exp_A_delta_t = np.zeros([2,2,T+1])
y = np.zeros(T+1)
# avg number of jumps/unit time
c = 10

b_M=alpha/(alpha-1)*c**((alpha-1)/alpha)

Z=np.array([1, 0, 0]) # observation matrix
C_v = kappa_V**2 # observation noise covariance

C_e = np.identity(3)

# generate data
for t in range(len(time_axis)-1):
    t+=1
    t_i = time_axis[t]
    delta_t_i=t_i-time_axis[t-1]

    # generate data
    X_true[:,t],drift_true[:,t],y[t],exp_A_delta_t[:,:,t],_,_,_,_,_=update_stable_langevin(X_true[:,t-1],drift_true[:,t-1],theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W, sigma_W, time_axis[t-1],v1,v2,M1,M2,M3,sigma_v)

# initial value
X_true2 = np.zeros([2,T+1])
drift_true2 = np.zeros([2,T+1])
exp_A_delta_t2 = np.zeros([2,2,T+1])
y2 = np.zeros(T+1)

# generate data
for t in range(len(time_axis)-1):
    t+=1
    t_i = time_axis[t]
    delta_t_i=t_i-time_axis[t-1]

    # generate data
    X_true2[:,t],drift_true2[:,t],y2[t],exp_A_delta_t2[:,:,t],_,_,_,_,_=update_stable_langevin(X_true2[:,t-1],drift_true2[:,t-1],theta,b_M,eA0,eA1,c,t_i,delta_t_i,alpha, mu_W, sigma_W, time_axis[t-1],v1,v2,M1,M2,M3,sigma_v)

plt.plot(X_true[0,:],X_true2[0,:])
plt.show()