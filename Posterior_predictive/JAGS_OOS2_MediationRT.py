# JAGS_OOS2_MediationRT.py - Performs out-of-sample prediction of new data using
#                            found in-sample posterior distributions as priors
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
# Record of Revisions
#
# Date            Programmers                         Descriptions of Change
# ====         ================                       ======================
# 03/07/17 Anna-Lena Schubert & Michael Nunez Converted from JAGS_OOS_Correlation2.py
# 03/08/17      Michael Nunez                  Change randsampLength for memory test
# 03/10/17      Michael Nunez          Fixed dimensions of a and ter initializations
# 03/14/17      Michael Nunez             Change chain lengths to singleton dimensions
# 03/17/17      Michael Nunez                  Change randsampLength for memory test
# 08/30/17      Michael Nunez                  Increase randsampLength
# 03/09/18      Michael Nunez            Evaluate joint data
# 05/24/18      Michael Nunez                          Load local results
# 11/26/18      Michael Nunez                Converted from JAGS_OOS2_MediationRT.py
###UNFINISHED###

# Imports
import numpy as np
import numpy.ma as ma
import scipy.io as sio
import pyjags
from scipy import stats
import os
from time import strftime

# Data locations
saveloc = '/home/michael/data10/michael/intel'

# Load the posterior samples

# Approximate posterior distributions with distribution shapes of
# the original priors

chainLength = 30000  # chain length
results = sio.loadmat('../Results/Joint_MediationRT.mat')

# define the model
model = '''
model {

  for (m in 1:possamps) {
  for (i in 1:I) {
    y[m,i] ~ dnorm(RT_mu[m,task[i],person[i]],
                   RTy_invtheta[m,task[i],person[i]])
  }

  for(i in 1:N) {

    for(j in 1:6) {
      IQ[m,i,j] ~ dnorm(IQ_mu[m,j,i],IQ_invtheta[m,j])
    }
    for(j in 1:18) {
      ERPdata[m,i,j] ~ dnorm(ERP_mu[m,i,j],ERP_invtheta[m,j])
    }

    # IQ manifest variables
    IQ_mu[m,1,i] <- IQ_lambda[m,1]*IQ_eta[m,i]
    IQ_mu[m,2,i] <- IQ_lambda[m,2]*IQ_eta[m,i]
    IQ_mu[m,3,i] <- IQ_lambda[m,3]*IQ_eta[m,i]
    IQ_mu[m,4,i] <- IQ_lambda[m,4]*IQ_eta[m,i]
    IQ_mu[m,5,i] <- IQ_lambda[m,5]*IQ_eta[m,i]
    IQ_mu[m,6,i] <- IQ_lambda[m,6]*IQ_eta[m,i]

    # IQ latent variables
    IQ_eta[m,i] ~ dnorm(IQ_mu_eta[m,i], IQ_invpsi[m])
    IQ_mu_eta[m,i] <- beta[m,2]*RT_eta[m,i,1] + beta[m,3]*ERP_eta[m,i,1]

    #ERP manifest variables
    ERP_mu[m,i,1] <- ERP_lambda[m,1]*ERP_eta[m,i,2]
    ERP_mu[m,i,2] <- ERP_lambda[m,2]*ERP_eta[m,i,2]
    ERP_mu[m,i,3] <- ERP_lambda[m,3]*ERP_eta[m,i,2]
    ERP_mu[m,i,4] <- ERP_lambda[m,4]*ERP_eta[m,i,2]
    ERP_mu[m,i,5] <- ERP_lambda[m,5]*ERP_eta[m,i,2]
    ERP_mu[m,i,6] <- ERP_lambda[m,6]*ERP_eta[m,i,2]
    ERP_mu[m,i,7] <- ERP_lambda[m,7]*ERP_eta[m,i,3]
    ERP_mu[m,i,8] <- ERP_lambda[m,8]*ERP_eta[m,i,3]
    ERP_mu[m,i,9] <- ERP_lambda[m,9]*ERP_eta[m,i,3]
    ERP_mu[m,i,10] <- ERP_lambda[m,10]*ERP_eta[m,i,3]
    ERP_mu[m,i,11] <- ERP_lambda[m,11]*ERP_eta[m,i,3]
    ERP_mu[m,i,12] <- ERP_lambda[m,12]*ERP_eta[m,i,3]
    ERP_mu[m,i,13] <- ERP_lambda[m,13]*ERP_eta[m,i,4]
    ERP_mu[m,i,14] <- ERP_lambda[m,14]*ERP_eta[m,i,4]
    ERP_mu[m,i,15] <- ERP_lambda[m,15]*ERP_eta[m,i,4]
    ERP_mu[m,i,16] <- ERP_lambda[m,16]*ERP_eta[m,i,4]
    ERP_mu[m,i,17] <- ERP_lambda[m,17]*ERP_eta[m,i,4]
    ERP_mu[m,i,18] <- ERP_lambda[m,18]*ERP_eta[m,i,4]

    # ERP latent variables
    ERP_eta[m,i,1] ~ dnorm(ERP_mu_eta[m,i,1], ERP_invpsi[m,1])
    ERP_eta[m,i,2] ~ dnorm(ERP_mu_eta[m,i,2], ERP_invpsi[m,2])
    ERP_eta[m,i,3] ~ dnorm(ERP_mu_eta[m,i,3], ERP_invpsi[m,3])
    ERP_eta[m,i,4] ~ dnorm(ERP_mu_eta[m,i,4], ERP_invpsi[m,4])
    ERP_mu_eta[m,i,1] <- 0
    ERP_mu_eta[m,i,2] <- ERP_lambda[m,19]*ERP_eta[m,i,1]
    ERP_mu_eta[m,i,3] <- ERP_lambda[m,20]*ERP_eta[m,i,1]
    ERP_mu_eta[m,i,4] <- ERP_lambda[m,21]*ERP_eta[m,i,1]

    # Single-task reaction times
    RT_mu[m,1,i] ~ dnorm(RT_nu[m,1] + RT_lambda[m,1]*RT_eta[m,i,2], RT_invtheta[m,1])
    RT_mu[m,2,i] ~ dnorm(RT_nu[m,2] + RT_lambda[m,2]*RT_eta[m,i,2], RT_invtheta[m,2])
    RT_mu[m,3,i] ~ dnorm(RT_nu[m,3] + RT_lambda[m,3]*RT_eta[m,i,3], RT_invtheta[m,3])
    RT_mu[m,4,i] ~ dnorm(RT_nu[m,4] + RT_lambda[m,4]*RT_eta[m,i,3], RT_invtheta[m,4])
    RT_mu[m,5,i] ~ dnorm(RT_nu[m,5] + RT_lambda[m,5]*RT_eta[m,i,3], RT_invtheta[m,5])
    RT_mu[m,6,i] ~ dnorm(RT_nu[m,6] + RT_lambda[m,6]*RT_eta[m,i,4], RT_invtheta[m,6])
    RT_mu[m,7,i] ~ dnorm(RT_nu[m,7] + RT_lambda[m,7]*RT_eta[m,i,4], RT_invtheta[m,7])
    RT_mu[m,8,i] ~ dnorm(RT_nu[m,8] + RT_lambda[m,8]*RT_eta[m,i,2], RT_invtheta[m,8])
    RT_mu[m,9,i] ~ dnorm(RT_nu[m,9] + RT_lambda[m,9]*RT_eta[m,i,2], RT_invtheta[m,9])
    RT_mu[m,10,i] ~ dnorm(RT_nu[m,10] + RT_lambda[m,10]*RT_eta[m,i,3], RT_invtheta[m,10])
    RT_mu[m,11,i] ~ dnorm(RT_nu[m,11] + RT_lambda[m,11]*RT_eta[m,i,3], RT_invtheta[m,11])
    RT_mu[m,12,i] ~ dnorm(RT_nu[m,12] + RT_lambda[m,12]*RT_eta[m,i,3], RT_invtheta[m,12])
    RT_mu[m,13,i] ~ dnorm(RT_nu[m,13] + RT_lambda[m,13]*RT_eta[m,i,4], RT_invtheta[m,13])
    RT_mu[m,14,i] ~ dnorm(RT_nu[m,14] + RT_lambda[m,14]*RT_eta[m,i,4], RT_invtheta[m,14])

    #Reaction time latent variables
    RT_eta[m,i,1] ~ dnorm(RT_mu_eta[m,i,1],RT_psi[m,1])
    RT_eta[m,i,2] ~ dnorm(RT_mu_eta[m,i,2],RT_psi[m,2])
    RT_eta[m,i,3] ~ dnorm(RT_mu_eta[m,i,3],RT_psi[m,3])
    RT_eta[m,i,4] ~ dnorm(RT_mu_eta[m,i,4],RT_psi[m,4])
    RT_mu_eta[m,i,1] <- beta[m,1]*ERP_eta[m,i,1]
    RT_mu_eta[m,i,2] <- RT_lambda[m,15]*RT_eta[m,i,1]
    RT_mu_eta[m,i,3] <- RT_lambda[m,16]*RT_eta[m,i,1]
    RT_mu_eta[m,i,4] <- RT_lambda[m,17]*RT_eta[m,i,1]

    #Reaction time variances
    for (t in 1:T) {
            RTy_invtheta[m,t,i] <- 1/RTy_theta[m,t] #Use mean task data
         }
    }

  # IQ Variances
  for(j in 1:6) {
      IQ_invtheta[m,j] <- 1/IQ_theta[m,j]
  }

  # ERP Variances
  for(j in 1:18) {
      ERP_invtheta[m,j] <- 1/ERP_theta[m,j]
  }

  # Reaction time Variances
  for(t in 1:T) {
      RT_invtheta[m,t] <- 1/RT_theta[m,t]
  }

  # Latent IQ Variances
  IQ_invpsi[m] <- 1/IQ_psi[m]

  # Latent ERP and reactiont time Variances
  for(j in 1:4) {
      ERP_invpsi[m,j] <- 1/ERP_psi[m,j]
      RT_invpsi[m,j] <- 1/RT_psi[m,j]
  }
    }


} # End of model
'''

np.random.seed(13)
for t in range(0,10):
  # find a random sample of posterior samples
  randsampLength = 50
  randsamp = np.random.permutation(chainLength)[0:randsampLength]

  # load estimated factor loadings

  IQ_lambda = np.empty([randsampLength, 6])
  IQ_theta = np.empty([randsampLength, 6])
  IQ_psi = np.empty([randsampLength, 1])
  ERP_lambda = np.empty([randsampLength, 21])
  ERP_theta = np.empty([randsampLength, 18])
  ERP_psi = np.empty([randsampLength, 4])
  RT_nu = np.empty([randsampLength, 14])
  RT_lambda = np.empty([randsampLength, 17])
  RT_theta = np.empty([randsampLength, 14])
  RTy_theta = np.empty([randsampLength, 14])
  RT_psi = np.empty([randsampLength, 4])
  beta = np.empty([randsampLength, 3])

  for i in range(0, 6):
      IQ_lambda[:, i] = results['chains'][0][0][
          'IQlambda_%d' % (i + 1)].reshape(chainLength)[randsamp]


  for i in range(0, 6):
      IQ_theta[:, i] = results['chains'][0][0][
          'IQtheta_%d' % (i + 1)].reshape(chainLength)[randsamp]


  for i in range(0, 21):
      ERP_lambda[:, i] = results['chains'][0][0][
          'ERPlambda_%d' % (i + 1)].reshape(chainLength)[randsamp]


  for i in range(0, 18):
      ERP_theta[:, i] = results['chains'][0][0][
          'ERPtheta_%d' % (i + 1)].reshape(chainLength)[randsamp]

  for i in range(0, 4):
      ERP_psi[:, i] = results['chains'][0][0][
          'ERPpsi_%d' % (i + 1)].reshape(chainLength)[randsamp]

  for i in range(0, 14):
      RT_nu[:, i] = results['chains'][0][0][
          'RTnu_%d' % (i + 1)].reshape(chainLength)[randsamp]


  for i in range(0, 17):
      RT_lambda[:, i] = results['chains'][0][0][
          'RTlambda_%d' % (i + 1)].reshape(chainLength)[randsamp]


  for i in range(0, 14):
      RT_theta[:, i] = results['chains'][0][0][
          'RTtheta_%d' % (i + 1)].reshape(chainLength)[randsamp]
      tempsamps = np.empty((randsampLength,92))
      for p in range(0,92):
        tempsamps[:,p] = results['chains'][0][0][
            'RTytheta_%d_%d' % (i + 1, p + 1)].reshape(chainLength)[randsamp]
      RTy_theta[:,i] = np.mean(tempsamps,axis=1)


  for i in range(0, 4):
      RT_psi[:, i] = results['chains'][0][0][
          'RTpsi_%d' % (i + 1)].reshape(chainLength)[randsamp]

  for i in range(0, 3):
    beta[:, i] = results['chains'][0][0][
        'beta_%d' % (i + 1)].reshape(chainLength)[randsamp]

  IQ_psi = results['chains'][0][0]['IQpsi'].reshape(chainLength)[randsamp]

  # load third data set
  erpiq = sio.loadmat('../Data/ThirdData.mat')
  rtacc = sio.loadmat('../Data/ThirdRT.mat')
  IQdata = erpiq['ThirdSet'][:, 1: 7]
  ERPdata = erpiq['ThirdSet'][:, 7: 25]
  N = ERPdata.shape[0] # number of participants

  # DO NOT replace missing values, use mask to let JAGS generate values
  ERPdata = ma.masked_array(ERPdata, mask=~(np.isfinite(ERPdata)))
  IQdata = ma.masked_array(IQdata, mask=~(np.isfinite(IQdata)))

  subjects = np.unique(rtacc['person'])
  P = len(subjects)

  for s in range(0, P):
      x = np.sum(rtacc['person'] == subjects[s])
      ID = np.ones([x, 1]) * (s + 1)
      if s == 0:
          IDs = ID
      else:
          IDs = np.vstack((IDs, ID))

  person = IDs
  T = np.max(rtacc['task'])
  I = len(rtacc['y'])
  F = 4
  y = np.abs(rtacc['y'])
  task = rtacc['task']
  # what is F?

  #Squeeze two dimensional vector data
  y = np.squeeze(y)
  person = np.squeeze(person)
  task = np.squeeze(task)
  y = ma.masked_array(y, mask=~(np.isfinite(y)))

  # pyjags code
  # Make sure $LD_LIBRARY_PATH sees /usr/local/lib
  pyjags.modules.load_module('wiener')
  pyjags.modules.load_module('dic')
  pyjags.modules.list_modules()

  nchains = 3
  burnin = 1  # Note that scientific notation breaks pyjags
  nsamps = 1

  # Track these variables
  trackvars = ['IQ_mu', 'IQ_lambda', 'IQ_theta', 'IQ_eta',
               'ERP_mu', 'ERP_lambda', 'ERP_theta', 'ERP_eta',
               'RT_nu', 'RT_lambda', 'RT_eta', 'RT_theta', 'RTy_theta',
               'beta', 'IQ', 'ERPdata', 'y']


  # Model name
  modelname = 'OOS2_%s_MediationRT_%s'
  rmnames = ['noIQ', 'noERP', 'noAccRT']

  # Create arrays large enough to contain predicted data
  IQlarge = np.tile(IQdata[None, ...], [randsampLength, 1, 1])
  ERPlarge = np.tile(ERPdata[None, ...], [randsampLength, 1, 1])
  ylarge = np.tile(y[None, ...], [randsampLength, 1])

  for m in range(0, 3):
      # Run JAGS model

      # Choose JAGS model type
      timestart = strftime('%b') + '_' + strftime('%d') + '_' + \
        strftime('%y') + '_' + strftime('%H') + '_' + strftime('%M')
      thismodel = (modelname % (rmnames[m], timestart))
      print 'Finding posterior predictives with model %s ...' % thismodel

      if m == 0:
          threaded = pyjags.Model(code=model,
                                  data=dict(IQ=ma.masked_array(IQlarge, mask=np.ones(IQlarge.size, dtype=bool)),
                                            ERPdata=ERPlarge, N=N, y=ylarge, person=person, task=task, T=T, I=I,
                                            possamps=randsampLength, IQ_lambda=IQ_lambda, IQ_theta=IQ_theta,
                                            ERP_lambda=ERP_lambda, ERP_theta=ERP_theta, RT_nu=RT_nu,
                                            ERP_psi=ERP_psi, IQ_psi=IQ_psi, RT_psi=RT_psi, RTy_theta=RTy_theta,
                                            RT_lambda=RT_lambda, RT_theta=RT_theta, beta=beta),
                                  chains=nchains, adapt=burnin, threads=nchains, progress_bar=True)
      elif m == 1:
          threaded = pyjags.Model(code=model,
                                  data=dict(ERPdata=ma.masked_array(ERPlarge, mask=np.ones(ERPlarge.size, dtype=bool)),
                                            IQ=IQlarge, N=N, y=ylarge, person=person, task=task, T=T, I=I,
                                            possamps=randsampLength, IQ_lambda=IQ_lambda, IQ_theta=IQ_theta,
                                            ERP_lambda=ERP_lambda, ERP_theta=ERP_theta, RT_nu=RT_nu,
                                            ERP_psi=ERP_psi, IQ_psi=IQ_psi, RT_psi=RT_psi, RTy_theta=RTy_theta,
                                            RT_lambda=RT_lambda, RT_theta=RT_theta, beta=beta),
                                  chains=nchains, adapt=burnin, threads=nchains, progress_bar=True)
      elif m == 2:
          threaded = pyjags.Model(code=model,
                                  data=dict(y=ma.masked_array(ylarge, mask=np.ones(ylarge.size, dtype=bool)),
                                            IQ=IQlarge, N=N, ERPdata=ERPlarge, person=person, task=task, T=T, I=I,
                                            possamps=randsampLength, IQ_lambda=IQ_lambda, IQ_theta=IQ_theta,
                                            ERP_lambda=ERP_lambda, ERP_theta=ERP_theta, RT_nu=RT_nu,
                                            ERP_psi=ERP_psi, IQ_psi=IQ_psi, RT_psi=RT_psi, RTy_theta=RTy_theta,
                                            RT_lambda=RT_lambda, RT_theta=RT_theta, beta=beta),
                                  chains=nchains, adapt=burnin, threads=nchains, progress_bar=True)

      samples = threaded.sample(nsamps, vars=trackvars, thin=10)

      savestring = saveloc + '/jagsout/behavmodel_' + \
          thismodel + ".mat"

      print 'Saving %s results to: \n %s' % (thismodel, savestring)

      sio.savemat(savestring, samples)