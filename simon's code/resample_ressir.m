

% Residual resampling by Liu & Chen JASA 1998

% actually originally developed by J. Baker Conf. Genetic Algorithms 1985 

% better (in terms of variance) and quicker than SIR

% N. Bergman/A. Doucet 1998



% Contrary to SIR, it requires to resample exactly N=length(weights) particles



function [index]=resample_ressir(weights)

N=length(weights);

t0=clock;

% first integer part

wn_res=N.*weights;

N_sons=fix(wn_res);


% residual number of particles to sample

N_first=sum(N_sons);

N_res=N-N_first;

index=zeros(1,N);

% put the index

ind=1;

for j=1:N

   if (N_sons(1,j)>0)

      index(1,ind:(ind+N_sons(1,j)-1))=j;

      ind=ind+N_sons(1,j);

   end

end

if (N_res~=0)

   

   wn_res=(wn_res-N_sons)/N_res;

   

   % generate the cumulative distribution

   dist=cumsum(wn_res);

   

   % generate N_res ordered random variables uniformly distributed in [0,1]

   u = fliplr(cumprod(rand(1,N_res).^(1./(N_res:-1:1))));

   

   j=1;

   for i=1:N_res

      while (u(1,i)>dist(1,j))

         j=j+1;

      end

      index(1,i+N_first)=j;

   end

   

end

