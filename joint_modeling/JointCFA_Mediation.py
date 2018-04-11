#JointCFA_Mediation.m     Contains all three CFAs (IQ, ERP, Diffusion
#               model) with the mediation connector
#

# Copyright (C) 2018 Anna-Lena Schubert <anna-lena.schubert@psychologie.uni-heidelberg.de>
#                    & Michael D. Nunez <mdnunez1@uci.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

##Record of revisions:
#   Date           Programmers               Description of change
#   ====        =================            =====================
#  02/20/17      Michael Nunez               Converted from JointCFA_Mediation.m

## Imports
import numpy as np
import numpy.ma as ma
import scipy.io as sio
import pyjags
from scipy import stats
import os

## Load the data

firstset = sio.loadmat('../Data/data.mat');        # load IQ and ERP data
secondset = sio.loadmat('../Data/SecondData.mat');
firstsetRT = sio.loadmat('../Data/RTdata.mat');      # load RT data
secondsetRT = sio.loadmat('../Data/SecondRT.mat');

bothdata = np.concatenate((firstset['data'] , secondset['SecondSet']),axis=0)
N = np.shape(bothdata)[0]  # number of participants
IQdata = bothdata[:,1:7]  # IQ data
ERPdata = bothdata[:,7:25] # ERP data

person = np.concatenate((firstsetRT['person'] , secondsetRT['person']),axis=0).squeeze()
task = np.concatenate((firstsetRT['task'] , secondsetRT['task']),axis=0).squeeze()
y = np.concatenate((firstsetRT['y'] , secondsetRT['y']),axis=0).squeeze()

subjects = np.unique(person);
P = np.shape(subjects)[0];
IDs = np.array([],dtype='int16');

subtrack = 1
for s in subjects:
  x = np.sum(person==s);
  ID = np.array(np.ones((x))*subtrack,dtype='int16');
  IDs = np.concatenate((IDs , ID),axis=0) if IDs.size else ID
  subtrack += 1


person = IDs;
T = np.max(task);
I = y.shape[0];

## IQ+ERP+Diffusion model

# define the model
model = '''
model {
  for (i in 1:I) {
    y[i] ~ dwiener(a[task[i],person[i]],
                   ter[task[i],person[i]],
                   0.5,
                   v[task[i],person[i]])
  }
  for(i in 1:N) {
    
    for(j in 1:6) {
      IQ[i,j] ~ dnorm(IQ_mu[j,i],IQ_invtheta[j])
    }
    for(j in 1:18) {
      ERPdata[i,j] ~ dnorm(ERP_mu[i,j],ERP_invtheta[j])
    }

    # IQ manifest variables
    IQ_mu[1,i] <- IQ_lambda[1]*IQ_eta[i]
    IQ_mu[2,i] <- IQ_lambda[2]*IQ_eta[i]
    IQ_mu[3,i] <- IQ_lambda[3]*IQ_eta[i]
    IQ_mu[4,i] <- IQ_lambda[4]*IQ_eta[i]
    IQ_mu[5,i] <- IQ_lambda[5]*IQ_eta[i]
    IQ_mu[6,i] <- IQ_lambda[6]*IQ_eta[i]

    # IQ latent variables
    IQ_eta[i] ~ dnorm(IQ_mu_eta[i], IQ_invpsi)
    IQ_mu_eta[i] <- beta[2]*v_eta[i,1] + beta[3]*ERP_eta[i,1]
   
    #ERP manifest variables
    ERP_mu[i,1] <- ERP_lambda[1]*ERP_eta[i,2]
    ERP_mu[i,2] <- ERP_lambda[2]*ERP_eta[i,2]
    ERP_mu[i,3] <- ERP_lambda[3]*ERP_eta[i,2]
    ERP_mu[i,4] <- ERP_lambda[4]*ERP_eta[i,2]
    ERP_mu[i,5] <- ERP_lambda[5]*ERP_eta[i,2]
    ERP_mu[i,6] <- ERP_lambda[6]*ERP_eta[i,2]
    ERP_mu[i,7] <- ERP_lambda[7]*ERP_eta[i,3]
    ERP_mu[i,8] <- ERP_lambda[8]*ERP_eta[i,3]
    ERP_mu[i,9] <- ERP_lambda[9]*ERP_eta[i,3]
    ERP_mu[i,10] <- ERP_lambda[10]*ERP_eta[i,3]
    ERP_mu[i,11] <- ERP_lambda[11]*ERP_eta[i,3]
    ERP_mu[i,12] <- ERP_lambda[12]*ERP_eta[i,3]
    ERP_mu[i,13] <- ERP_lambda[13]*ERP_eta[i,4]
    ERP_mu[i,14] <- ERP_lambda[14]*ERP_eta[i,4]
    ERP_mu[i,15] <- ERP_lambda[15]*ERP_eta[i,4]
    ERP_mu[i,16] <- ERP_lambda[16]*ERP_eta[i,4]
    ERP_mu[i,17] <- ERP_lambda[17]*ERP_eta[i,4]
    ERP_mu[i,18] <- ERP_lambda[18]*ERP_eta[i,4]
   
    # ERP latent variables
    ERP_eta[i,1] ~ dnorm(ERP_mu_eta[i,1], ERP_invpsi[1])
    ERP_eta[i,2] ~ dnorm(ERP_mu_eta[i,2], ERP_invpsi[2])
    ERP_eta[i,3] ~ dnorm(ERP_mu_eta[i,3], ERP_invpsi[3])
    ERP_eta[i,4] ~ dnorm(ERP_mu_eta[i,4], ERP_invpsi[4])
    ERP_mu_eta[i,1] <- 0
    ERP_mu_eta[i,2] <- ERP_lambda[19]*ERP_eta[i,1]
    ERP_mu_eta[i,3] <- ERP_lambda[20]*ERP_eta[i,1]
    ERP_mu_eta[i,4] <- ERP_lambda[21]*ERP_eta[i,1]
    
    # Single-task drift rates
    v[1,i] ~ dnorm(v_nu[1] + v_lambda[1]*v_eta[i,2], v_invtheta[1]) 
    v[2,i] ~ dnorm(v_nu[2] + v_lambda[2]*v_eta[i,2], v_invtheta[2])
    v[3,i] ~ dnorm(v_nu[3] + v_lambda[3]*v_eta[i,3], v_invtheta[3])
    v[4,i] ~ dnorm(v_nu[4] + v_lambda[4]*v_eta[i,3], v_invtheta[4])
    v[5,i] ~ dnorm(v_nu[5] + v_lambda[5]*v_eta[i,3], v_invtheta[5])
    v[6,i] ~ dnorm(v_nu[6] + v_lambda[6]*v_eta[i,4], v_invtheta[6])
    v[7,i] ~ dnorm(v_nu[7] + v_lambda[7]*v_eta[i,4], v_invtheta[7])
    v[8,i] ~ dnorm(v_nu[8] + v_lambda[8]*v_eta[i,2], v_invtheta[8])
    v[9,i] ~ dnorm(v_nu[9] + v_lambda[9]*v_eta[i,2], v_invtheta[9])
    v[10,i] ~ dnorm(v_nu[10] + v_lambda[10]*v_eta[i,3], v_invtheta[10])
    v[11,i] ~ dnorm(v_nu[11] + v_lambda[11]*v_eta[i,3], v_invtheta[11])
    v[12,i] ~ dnorm(v_nu[12] + v_lambda[12]*v_eta[i,3], v_invtheta[12])
    v[13,i] ~ dnorm(v_nu[13] + v_lambda[13]*v_eta[i,4], v_invtheta[13])
    v[14,i] ~ dnorm(v_nu[14] + v_lambda[14]*v_eta[i,4], v_invtheta[14])
    
    #Drift rate latent variables
    v_eta[i,1] ~ dnorm(v_mu_eta[i,1],v_psi[1])
    v_eta[i,2] ~ dnorm(v_mu_eta[i,2],v_psi[2])
    v_eta[i,3] ~ dnorm(v_mu_eta[i,3],v_psi[3])
    v_eta[i,4] ~ dnorm(v_mu_eta[i,4],v_psi[4])
    v_mu_eta[i,1] <- beta[1]*ERP_eta[i,1]
    v_mu_eta[i,2] <- v_lambda[15]*v_eta[i,1]
    v_mu_eta[i,3] <- v_lambda[16]*v_eta[i,1]
    v_mu_eta[i,4] <- v_lambda[17]*v_eta[i,1]
    
    for (t in 1:T) {
            a[t,i]   ~ dnorm(1.0, pow(0.5, -2.0)) T(0, 5)
            ter[t,i] ~ dnorm(0.3, pow(0.2, -2.0)) T(0, 1)
         }
    }

  # Common factor loadings
  beta[1] ~ dnorm(0,1e-2)
  beta[2] ~ dnorm(0,1e-2)
  beta[3] ~ dnorm(0,1e-2)
  
  # IQ Loadings
  IQ_lambda[1] <- 1
  for (l in 2:6) {
  IQ_lambda[l] ~ dnorm(0,1e-2)T(0,)
  }

  # IQ Precision
  IQ_invtheta[1] ~ dgamma(1,.5)   # precision PC
  IQ_invtheta[2] ~ dgamma(1,.5)   # precision PS
  IQ_invtheta[3] ~ dgamma(1,.5)   # precision M
  IQ_invtheta[4] ~ dgamma(1,.5)   # precision C
  IQ_invtheta[5] ~ dgamma(1,.5)   # precision APModd
  IQ_invtheta[6] ~ dgamma(1,.5)   # precision APMeven
  IQ_invpsi ~ dgamma(1,.5)  # precision g
  
  # IQ Variances
  
  for(j in 1:6) {
      IQ_theta[j] <- 1/IQ_invtheta[j]
  }
  IQ_psi <- 1/IQ_invpsi

  # ERP Loadings
   
  ERP_lambda[1] <- 1
  ERP_lambda[7] <- 1
  ERP_lambda[13] <- 1
  ERP_lambda[19] <- 1
  ERP_lambda[20] ~ dnorm(1,pow(1,-2))T(0,)
  ERP_lambda[21] ~ dnorm(1,pow(1,-2))T(0,)
  for (l in 2:6){
  ERP_lambda[l] ~ dnorm(1,pow(1,-2))T(0,)
  }
  for (l in 8:12){
  ERP_lambda[l] ~ dnorm(1,pow(1,-2))T(0,)
  }
  for (l in 14:18){
  ERP_lambda[l] ~ dnorm(1,pow(1,-2))T(0,)
  }
  
  # ERP Precision
   
  ERP_invtheta[1] ~ dgamma(1,.5)   # precision CRTP2S1
  ERP_invtheta[2] ~ dgamma(1,.5)   # precision SP2S1
  ERP_invtheta[3] ~ dgamma(1,.5)   # precision PP2S1
  ERP_invtheta[4] ~ dgamma(1,.5)   # precision CRTP2S2
  ERP_invtheta[5] ~ dgamma(1,.5)   # precision SP2S2
  ERP_invtheta[6] ~ dgamma(1,.5)   # precision PP2S2
  ERP_invtheta[7] ~ dgamma(1,.5)   # precision CRTN2S1
  ERP_invtheta[8] ~ dgamma(1,.5)   # precision SN2S1
  ERP_invtheta[9] ~ dgamma(1,.5)   # precision PN2S1
  ERP_invtheta[10] ~ dgamma(1,.5)  # precision CRTN2S2
  ERP_invtheta[11] ~ dgamma(1,.5)  # precision SN2S2
  ERP_invtheta[12] ~ dgamma(1,.5)  # precision PN2S2
  ERP_invtheta[13] ~ dgamma(1,.5)  # precision CRTP3S1
  ERP_invtheta[14] ~ dgamma(1,.5)  # precision SP3S1
  ERP_invtheta[15] ~ dgamma(1,.5)  # precision PP3S1
  ERP_invtheta[16] ~ dgamma(1,.5)  # precision CRTP3S2
  ERP_invtheta[17] ~ dgamma(1,.5)  # precision SP3S2
  ERP_invtheta[18] ~ dgamma(1,.5)  # precision PP3S2
   
  ERP_invpsi[1] ~ dgamma(1,.5) 
  ERP_invpsi[2] ~ dgamma(1,.5)       # precision N2
  ERP_invpsi[3] ~ dgamma(1,.5)       # precision P3
  ERP_invpsi[4] ~ dgamma(1,.5)       # precision ERPLatencies
   
  # ERP Variances
  
  for(j in 1:18) {
      ERP_theta[j] <- 1/ERP_invtheta[j]
  }
  for (j in 1:4) {
  ERP_psi[j] <- 1/ERP_invpsi[j]
  }
    
  # Diffusion Intercepts
  v_nu[1] ~ dnorm(2,pow(1.5,-2))
  v_nu[2] ~ dnorm(2,pow(1.5,-2))
  v_nu[3] ~ dnorm(2,pow(1.5,-2))
  v_nu[4] ~ dnorm(2,pow(1.5,-2))
  v_nu[5] ~ dnorm(2,pow(1.5,-2))
  v_nu[6] ~ dnorm(2,pow(1.5,-2))
  v_nu[7] ~ dnorm(2,pow(1.5,-2))
  v_nu[8] ~ dnorm(2,pow(1.5,-2))
  v_nu[9] ~ dnorm(2,pow(1.5,-2))
  v_nu[10] ~ dnorm(2,pow(1.5,-2))
  v_nu[11] ~ dnorm(2,pow(1.5,-2))
  v_nu[12] ~ dnorm(2,pow(1.5,-2))
  v_nu[13] ~ dnorm(2,pow(1.5,-2))
  v_nu[14] ~ dnorm(2,pow(1.5,-2))
  
  
  # Diffusion Loadings
  v_lambda[1] <- 1
  v_lambda[2] ~ dnorm(1,pow(1,-2))T(0,)
  v_lambda[3] <- 1
  v_lambda[4] ~ dnorm(1,pow(1,-2))T(0,)
  v_lambda[5] ~ dnorm(1,pow(1,-2))T(0,)
  v_lambda[6] <- 1
  for (l in 7:16) {
  v_lambda[l] ~ dnorm(1,pow(1,-2))T(0,)
  }
  v_lambda[17] <- 1
  
  # Diffuion precision
  v_invtheta[1] ~ dgamma(1,.5)
  v_invtheta[2] ~ dgamma(1,.5)
  v_invtheta[3] ~ dgamma(1,.5)
  v_invtheta[4] ~ dgamma(1,.5)
  v_invtheta[5] ~ dgamma(1,.5)
  v_invtheta[6] ~ dgamma(1,.5)
  v_invtheta[7] ~ dgamma(1,.5)
  v_invtheta[8] ~ dgamma(1,.5)
  v_invtheta[9] ~ dgamma(1,.5)
  v_invtheta[10] ~ dgamma(1,.5)
  v_invtheta[11] ~ dgamma(1,.5)
  v_invtheta[12] ~ dgamma(1,.5)
  v_invtheta[13] ~ dgamma(1,.5)
  v_invtheta[14] ~ dgamma(1,.5)
  
  v_invpsi[1] ~ dgamma(1,.5)
  v_invpsi[2] ~ dgamma(1,.5)
  v_invpsi[3] ~ dgamma(1,.5)
  v_invpsi[4] ~ dgamma(1,.5)
  
  # Diffuion variances
  for (t in 1:T) {
    v_theta[t] <- 1/v_invtheta[t]
  }
  for (j in 1:4) {
  v_psi[j] <- 1/v_invpsi[j]
  }
    
   

} # End of model
'''


## pyjags code

# Make sure $LD_LIBRARY_PATH sees /usr/local/lib
pyjags.modules.load_module('wiener')
pyjags.modules.load_module('dic')
pyjags.modules.list_modules()

nchains = 3
burnin = 1000  # Note that scientific notation breaks pyjags
nsamps = 10000
thin = 10 # Results in 3*(10000/10) = 3000 samples

# parameters of interest
trackvars = ['IQ_mu','IQ_lambda', 'IQ_theta', 'IQ_eta','IQ_psi', 
    'ERP_mu','ERP_lambda','ERP_theta','ERP_eta', 'ERP_psi',
    'v_nu', 'v_lambda', 'v_eta', 'v_theta', 'v_psi', 'v', 'a', 'ter',
    'beta']


# Model name
modelname = 'Mediation'

# Create dictionary of initial values
initials = []
for c in range(0, nchains):
    chaininit = {
        'IQ_lambda'  :  np.concatenate((np.array([np.nan]), np.random.uniform(size=5)),axis=0),
        'IQ_invtheta': np.random.uniform(1., 3., size=6),
        'IQ_invpsi': np.random.uniform(1., 3., size=1),
        'ERP_lambda'  : np.concatenate((np.array([np.nan]), np.random.uniform(size=5), np.array([np.nan]), np.random.uniform(size=5), np.array([np.nan]), np.random.uniform(size=5), np.array([np.nan]), np.random.uniform(size=2)),axis=0),
        'ERP_invtheta': np.random.uniform(1., 3., size=18),
        'ERP_invpsi': np.random.uniform(1., 3., size=4),
        'v_nu'   : np.random.uniform(size=T) ,
        'v_lambda'   : np.concatenate((np.array([np.nan]), np.random.uniform(size=1), np.array([np.nan]), np.random.uniform(size=1), np.random.uniform(size=1), np.array([np.nan]), np.random.uniform(size=10), np.array([np.nan])),axis=0),
        'v_invtheta' : np.random.uniform(1., 3., size=T),
        'v_invpsi' : np.random.uniform(1., 3., size=4),
        'a': np.random.uniform(.5, 1., size=(T, P)),
        'ter': np.random.uniform(.02, .12, size=(T, P)),
        'beta': np.random.uniform(0, 1., size=3)
    }
    initials.append(chaininit)




# Run JAGS model

# Choose JAGS model type
print 'Finding posterior predictives with %s model ...' % modelname

threaded = pyjags.Model(code=model, init=initials,
                        data=dict(IQ=IQdata,
                                  ERPdata=ERPdata, N=N, y=y, person=person, task=task, T=T, I=I),
                        chains=nchains, adapt=burnin, threads=nchains, progress_bar=True)


samples = threaded.sample(nsamps, vars=trackvars, thin=thin)

savestring = '../Results/' + \
    modelname + ".mat"

print 'Saving %s results to: \n %s' % (modelname, savestring)
S
sio.savemat(savestring, samples)
