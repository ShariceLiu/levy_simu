kappa = 0.6;
theta = 0.4;
sigma = 0.5;
T = 1;
a = 5;

episilon = 1e-5;
sigma_e2 = (sigma^2+kappa*theta^2)*(1-2*normcdf(-sqrt(episilon/kappa)))-sqrt(2*kappa*episilon)/sqrt(pi)*exp(-episilon/(2*kappa));

%sys = zpk([0.5],[1,1],1);
sys = tf([2,-1],[1, -2, 1]);
h = nyquistplot(sys);