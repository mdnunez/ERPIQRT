%JointCFA_Mediation.m     Contains all three CFAs (IQ, ERP, Diffusion
%               model) with the mediation connector
%

% Copyright (C) 2018 Anna-Lena Schubert <anna-lena.schubert@psychologie.uni-heidelberg.de>
%                    & Michael D. Nunez <mdnunez1@uci.edu>
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% Record of revisions:
%   Date           Programmers               Description of change
%   ====        =================            =====================
%  11/21/16     Anna-Lena Schubert               Original code
%  11/22/16     Michael Nunez              Combining the models
%  11/23/16     Anna-Lena Schubert         Adding mediation model
%                                          (ERP -> drift -> IQ)
%  11/28/16     Michael Nunez               Result diagnostics
%  11/30/16     Anna-Lena Schubert         Removed truncation from
%                                          regression parameters
%  03/07/17     Anna-Lena Schubert         Fixed lambdas instead of
%                                          variances
%  02/01/18     Michael Nunez            Add second set of data
%                                       Conversion to releasable code
%  02/07/18     Michael Nunez               Fixes
%  04/11/18     Michael Nunez              Reassert parallel option

%% Notes:
% 1) Fix gnu parallel by removing "--tollef" line from /etc/parallel/config
%    Reference: https://github.com/tseemann/nullarbor/issues/42


%% Load the data

firstset = load('../Data/data.mat');        % load IQ and ERP data
secondset = load('../Data/SecondData.mat');
firstsetRT = load('../Data/RTdata.mat');      % load RT data
secondsetRT = load('../Data/SecondRT.mat');

bothdata = [firstset.data ; secondset.SecondSet];
N = numel(bothdata(:,1));   % number of participants
IQdata = bothdata(:,2:7);   % IQ data
ERPdata = bothdata(:,8:25); % ERP data

person = [firstsetRT.person ; secondsetRT.person];
task = [firstsetRT.task ; secondsetRT.task];
y = [firstsetRT.y ; secondsetRT.y];

subjects = unique(person);
P = length(subjects);
IDs = [];

for s = 1:P
    x = sum(person==subjects(s));
    ID = ones(x,1)*s;
    IDs = [IDs; ID];
end

person = IDs;
T = max(task);
I = numel(y);
F = 4;

data = struct(...
    'IQ', IQdata, ...
    'ERPdata', ERPdata, ...
    'N', N, ...
    'y'      ,      y , ...
    'person' , person , ...
    'task'   ,   task , ...
    'F'      ,      F , ...
    'T'      ,      T , ...
    'P'      ,      P , ...
    'I'      ,      I );

%% IQ+ERP+Diffusion model

% define the model
model = {
'model {'
'  for (i in 1:I) {'
'    y[i] ~ dwiener(a[task[i],person[i]],'
'                   ter[task[i],person[i]],'
'                   0.5,'
'                   v[task[i],person[i]])'
'  }'
'  for(i in 1:N) {'
'    '
'    for(j in 1:6) {'
'      IQ[i,j] ~ dnorm(IQ.mu[j,i],IQ.invtheta[j])'
'    }'
'    for(j in 1:18) {'
'      ERPdata[i,j] ~ dnorm(ERP.mu[i,j],ERP.invtheta[j])'
'    }'
''
'    # IQ manifest variables'
'    IQ.mu[1,i] <- IQ.lambda[1]*IQ.eta[i]'
'    IQ.mu[2,i] <- IQ.lambda[2]*IQ.eta[i]'
'    IQ.mu[3,i] <- IQ.lambda[3]*IQ.eta[i]'
'    IQ.mu[4,i] <- IQ.lambda[4]*IQ.eta[i]'
'    IQ.mu[5,i] <- IQ.lambda[5]*IQ.eta[i]'
'    IQ.mu[6,i] <- IQ.lambda[6]*IQ.eta[i]'
''
'    # IQ latent variables'
'    IQ.eta[i] ~ dnorm(IQ.mu.eta[i], IQ.invpsi)'
'    IQ.mu.eta[i] <- beta[2]*v.eta[i,1] + beta[3]*ERP.eta[i,1]'
'   '
'    #ERP manifest variables'
'    ERP.mu[i,1] <- ERP.lambda[1]*ERP.eta[i,2]'
'    ERP.mu[i,2] <- ERP.lambda[2]*ERP.eta[i,2]'
'    ERP.mu[i,3] <- ERP.lambda[3]*ERP.eta[i,2]'
'    ERP.mu[i,4] <- ERP.lambda[4]*ERP.eta[i,2]'
'    ERP.mu[i,5] <- ERP.lambda[5]*ERP.eta[i,2]'
'    ERP.mu[i,6] <- ERP.lambda[6]*ERP.eta[i,2]'
'    ERP.mu[i,7] <- ERP.lambda[7]*ERP.eta[i,3]'
'    ERP.mu[i,8] <- ERP.lambda[8]*ERP.eta[i,3]'
'    ERP.mu[i,9] <- ERP.lambda[9]*ERP.eta[i,3]'
'    ERP.mu[i,10] <- ERP.lambda[10]*ERP.eta[i,3]'
'    ERP.mu[i,11] <- ERP.lambda[11]*ERP.eta[i,3]'
'    ERP.mu[i,12] <- ERP.lambda[12]*ERP.eta[i,3]'
'    ERP.mu[i,13] <- ERP.lambda[13]*ERP.eta[i,4]'
'    ERP.mu[i,14] <- ERP.lambda[14]*ERP.eta[i,4]'
'    ERP.mu[i,15] <- ERP.lambda[15]*ERP.eta[i,4]'
'    ERP.mu[i,16] <- ERP.lambda[16]*ERP.eta[i,4]'
'    ERP.mu[i,17] <- ERP.lambda[17]*ERP.eta[i,4]'
'    ERP.mu[i,18] <- ERP.lambda[18]*ERP.eta[i,4]'
'   '
'    # ERP latent variables'
'    ERP.eta[i,1] ~ dnorm(ERP.mu.eta[i,1], ERP.invpsi[1])'
'    ERP.eta[i,2] ~ dnorm(ERP.mu.eta[i,2], ERP.invpsi[2])'
'    ERP.eta[i,3] ~ dnorm(ERP.mu.eta[i,3], ERP.invpsi[3])'
'    ERP.eta[i,4] ~ dnorm(ERP.mu.eta[i,4], ERP.invpsi[4])'
'    ERP.mu.eta[i,1] <- 0'
'    ERP.mu.eta[i,2] <- ERP.lambda[19]*ERP.eta[i,1]'
'    ERP.mu.eta[i,3] <- ERP.lambda[20]*ERP.eta[i,1]'
'    ERP.mu.eta[i,4] <- ERP.lambda[21]*ERP.eta[i,1]'
'    '
'    # Single-task drift rates'
'    v[1,i] ~ dnorm(v.nu[1] + v.lambda[1]*v.eta[i,2], v.invtheta[1])' 
'    v[2,i] ~ dnorm(v.nu[2] + v.lambda[2]*v.eta[i,2], v.invtheta[2])'
'    v[3,i] ~ dnorm(v.nu[3] + v.lambda[3]*v.eta[i,3], v.invtheta[3])'
'    v[4,i] ~ dnorm(v.nu[4] + v.lambda[4]*v.eta[i,3], v.invtheta[4])'
'    v[5,i] ~ dnorm(v.nu[5] + v.lambda[5]*v.eta[i,3], v.invtheta[5])'
'    v[6,i] ~ dnorm(v.nu[6] + v.lambda[6]*v.eta[i,4], v.invtheta[6])'
'    v[7,i] ~ dnorm(v.nu[7] + v.lambda[7]*v.eta[i,4], v.invtheta[7])'
'    v[8,i] ~ dnorm(v.nu[8] + v.lambda[8]*v.eta[i,2], v.invtheta[8])'
'    v[9,i] ~ dnorm(v.nu[9] + v.lambda[9]*v.eta[i,2], v.invtheta[9])'
'    v[10,i] ~ dnorm(v.nu[10] + v.lambda[10]*v.eta[i,3], v.invtheta[10])'
'    v[11,i] ~ dnorm(v.nu[11] + v.lambda[11]*v.eta[i,3], v.invtheta[11])'
'    v[12,i] ~ dnorm(v.nu[12] + v.lambda[12]*v.eta[i,3], v.invtheta[12])'
'    v[13,i] ~ dnorm(v.nu[13] + v.lambda[13]*v.eta[i,4], v.invtheta[13])'
'    v[14,i] ~ dnorm(v.nu[14] + v.lambda[14]*v.eta[i,4], v.invtheta[14])'
'    '
'    #Drift rate latent variables'
'    v.eta[i,1] ~ dnorm(v.mu.eta[i,1],v.psi[1])'
'    v.eta[i,2] ~ dnorm(v.mu.eta[i,2],v.psi[2])'
'    v.eta[i,3] ~ dnorm(v.mu.eta[i,3],v.psi[3])'
'    v.eta[i,4] ~ dnorm(v.mu.eta[i,4],v.psi[4])'
'    v.mu.eta[i,1] <- beta[1]*ERP.eta[i,1]'
'    v.mu.eta[i,2] <- v.lambda[15]*v.eta[i,1]'
'    v.mu.eta[i,3] <- v.lambda[16]*v.eta[i,1]'
'    v.mu.eta[i,4] <- v.lambda[17]*v.eta[i,1]'
'    '
'    for (t in 1:T) {'
'            a[t,i]   ~ dnorm(1.0, pow(0.5, -2.0)) T(0, 5)'
'            ter[t,i] ~ dnorm(0.3, pow(0.2, -2.0)) T(0, 1)'
'         }'
'    }'
''
'  # Common factor loadings'
'  beta[1] ~ dnorm(0,1e-2)'
'  beta[2] ~ dnorm(0,1e-2)'
'  beta[3] ~ dnorm(0,1e-2)'
'  '
'  # IQ Loadings'
'  IQ.lambda[1] <- 1'
'  for (l in 2:6) {'
'  IQ.lambda[l] ~ dnorm(0,1e-2)T(0,)'
'  }'
''
'  # IQ Precision'
'  IQ.invtheta[1] ~ dgamma(1,.5)   # precision PC'
'  IQ.invtheta[2] ~ dgamma(1,.5)   # precision PS'
'  IQ.invtheta[3] ~ dgamma(1,.5)   # precision M'
'  IQ.invtheta[4] ~ dgamma(1,.5)   # precision C'
'  IQ.invtheta[5] ~ dgamma(1,.5)   # precision APModd'
'  IQ.invtheta[6] ~ dgamma(1,.5)   # precision APMeven'
'  IQ.invpsi ~ dgamma(1,.5)  # precision g'
'  '
'  # IQ Variances'
'  '
'  for(j in 1:6) {'
'      IQ.theta[j] <- 1/IQ.invtheta[j]'
'  }'
'  IQ.psi <- 1/IQ.invpsi'
''
'  # ERP Loadings'
'   '
'  ERP.lambda[1] <- 1'
'  ERP.lambda[7] <- 1'
'  ERP.lambda[13] <- 1'
'  ERP.lambda[19] <- 1'
'  ERP.lambda[20] ~ dnorm(1,pow(1,-2))T(0,)'
'  ERP.lambda[21] ~ dnorm(1,pow(1,-2))T(0,)'
'  for (l in 2:6){'
'  ERP.lambda[l] ~ dnorm(1,pow(1,-2))T(0,)'
'  }'
'  for (l in 8:12){'
'  ERP.lambda[l] ~ dnorm(1,pow(1,-2))T(0,)'
'  }'
'  for (l in 14:18){'
'  ERP.lambda[l] ~ dnorm(1,pow(1,-2))T(0,)'
'  }'
'  '
'  # ERP Precision'
'   '
'  ERP.invtheta[1] ~ dgamma(1,.5)   # precision CRTP2S1'
'  ERP.invtheta[2] ~ dgamma(1,.5)   # precision SP2S1'
'  ERP.invtheta[3] ~ dgamma(1,.5)   # precision PP2S1'
'  ERP.invtheta[4] ~ dgamma(1,.5)   # precision CRTP2S2'
'  ERP.invtheta[5] ~ dgamma(1,.5)   # precision SP2S2'
'  ERP.invtheta[6] ~ dgamma(1,.5)   # precision PP2S2'
'  ERP.invtheta[7] ~ dgamma(1,.5)   # precision CRTN2S1'
'  ERP.invtheta[8] ~ dgamma(1,.5)   # precision SN2S1'
'  ERP.invtheta[9] ~ dgamma(1,.5)   # precision PN2S1'
'  ERP.invtheta[10] ~ dgamma(1,.5)  # precision CRTN2S2'
'  ERP.invtheta[11] ~ dgamma(1,.5)  # precision SN2S2'
'  ERP.invtheta[12] ~ dgamma(1,.5)  # precision PN2S2'
'  ERP.invtheta[13] ~ dgamma(1,.5)  # precision CRTP3S1'
'  ERP.invtheta[14] ~ dgamma(1,.5)  # precision SP3S1'
'  ERP.invtheta[15] ~ dgamma(1,.5)  # precision PP3S1'
'  ERP.invtheta[16] ~ dgamma(1,.5)  # precision CRTP3S2'
'  ERP.invtheta[17] ~ dgamma(1,.5)  # precision SP3S2'
'  ERP.invtheta[18] ~ dgamma(1,.5)  # precision PP3S2'
'   '
'  ERP.invpsi[1] ~ dgamma(1,.5)' 
'  ERP.invpsi[2] ~ dgamma(1,.5)       # precision N2'
'  ERP.invpsi[3] ~ dgamma(1,.5)       # precision P3'
'  ERP.invpsi[4] ~ dgamma(1,.5)       # precision ERPLatencies'
'   '
'  # ERP Variances'
'  '
'  for(j in 1:18) {'
'      ERP.theta[j] <- 1/ERP.invtheta[j]'
'  }'
'  for (j in 1:4) {'
'  ERP.psi[j] <- 1/ERP.invpsi[j]'
'  }'
'    '
'  # Diffusion Intercepts'
'  v.nu[1] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[2] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[3] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[4] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[5] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[6] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[7] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[8] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[9] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[10] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[11] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[12] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[13] ~ dnorm(2,pow(1.5,-2))'
'  v.nu[14] ~ dnorm(2,pow(1.5,-2))'
'  '
'  '
'  # Diffusion Loadings'
'  v.lambda[1] <- 1'
'  v.lambda[2] ~ dnorm(1,pow(1,-2))T(0,)'
'  v.lambda[3] <- 1'
'  v.lambda[4] ~ dnorm(1,pow(1,-2))T(0,)'
'  v.lambda[5] ~ dnorm(1,pow(1,-2))T(0,)'
'  v.lambda[6] <- 1'
'  for (l in 7:16) {'
'  v.lambda[l] ~ dnorm(1,pow(1,-2))T(0,)'
'  }'
'  v.lambda[17] <- 1'
'  '
'  # Diffuion precision'
'  v.invtheta[1] ~ dgamma(1,.5)'
'  v.invtheta[2] ~ dgamma(1,.5)'
'  v.invtheta[3] ~ dgamma(1,.5)'
'  v.invtheta[4] ~ dgamma(1,.5)'
'  v.invtheta[5] ~ dgamma(1,.5)'
'  v.invtheta[6] ~ dgamma(1,.5)'
'  v.invtheta[7] ~ dgamma(1,.5)'
'  v.invtheta[8] ~ dgamma(1,.5)'
'  v.invtheta[9] ~ dgamma(1,.5)'
'  v.invtheta[10] ~ dgamma(1,.5)'
'  v.invtheta[11] ~ dgamma(1,.5)'
'  v.invtheta[12] ~ dgamma(1,.5)'
'  v.invtheta[13] ~ dgamma(1,.5)'
'  v.invtheta[14] ~ dgamma(1,.5)'
'  '
'  v.invpsi[1] ~ dgamma(1,.5)'
'  v.invpsi[2] ~ dgamma(1,.5)'
'  v.invpsi[3] ~ dgamma(1,.5)'
'  v.invpsi[4] ~ dgamma(1,.5)'
'  '
'  # Diffuion variances'
'  for (t in 1:T) {'
'    v.theta[t] <- 1/v.invtheta[t]'
'  }'
'  for (j in 1:4) {'
'  v.psi[j] <- 1/v.invpsi[j]'
'  }'
'    '
'   '
''
'} # End of model'
};


%% Set up JAGS

% Starting values
generator = @()struct(...
    'IQ_lambda'  , [NaN rand(1,5)], ...
    'IQ_invtheta',rand(1,6)* 2.0 + 1.00, ...
    'IQ_invpsi', rand(1,1)* 2.0 + 1.00, ...
    'ERP_lambda'  , [NaN rand(1,5) NaN rand(1,5) NaN rand(1,5) NaN rand(1,2)], ...
    'ERP_invtheta', rand(1,18)* 2.0 + 1.00, ...
    'ERP_invpsi', rand(1,4)* 2.0 + 1.00, ...
    'v_nu'   , rand(1,T) * 1.0 + 0.00  , ...
    'v_lambda'   , [NaN rand NaN rand rand NaN rand(1,10) NaN], ...
    'v_invtheta' , rand(1,T)* 2.0 + 1.00,  ...
    'v_invpsi' , rand(1,4)* 2.0 + 1.00, ...
    'a'        , rand(T, P) * 0.5 + 0.50  , ...
    'ter'      , rand(T, P) * 0.1 + 0.02, ...
    'beta'   , rand(1,3));    

% parameters of interest
params = {'IQ.mu','IQ.lambda', 'IQ.theta', 'IQ.eta','IQ.psi', ...
    'ERP.mu','ERP.lambda','ERP.theta','ERP.eta', 'ERP.psi',...
    'v.nu', 'v.lambda', 'v.eta', 'v.theta', 'v.psi', 'v', 'a', 'ter',...
    'beta'};

% Set trinity parameters
modelname  = 'Mediation';
nsamples   = 1e4;
nburnin    = 2e3;
nchains    =   3;
thin       =  10;
verbosity  =   2;
maxcores   =   3;
modules    = {'wiener' 'dic'};

try 
    trinity.assert_parallel()
    parallelit = true;
    disp 'Parallel detected, will use.'
catch me
    parallelit = false;
    disp 'Parallel failed, will perform sequential.'
end

% Tell Trinity which engine to use
engine = 'jags';

%% Run Trinity with the CALLBAYES() function

tic

[stats, chains, diagnostics, info] = callbayes(engine, ...
    'model'          ,     model , ...
    'data'           ,      data , ...
    'outputname'     , 'samples' , ...
    'init'           , generator , ...
    'modelfilename'  ,   modelname , ...
    'datafilename'   ,    modelname , ...
    'initfilename'   ,    modelname , ...
    'scriptfilename' ,    modelname , ...
    'logfilename'    ,    modelname , ...
    'modules'        ,     modules  , ...
    'nchains'        ,     nchains  , ...
    'nburnin'        ,     nburnin  , ...
    'nsamples'       ,    nsamples  , ...
    'monitorparams'  ,    params , ...
    'thin'           ,        thin  , ...
    'workingdir'     ,    ['/tmp/' modelname]  , ...
    'verbosity'      ,        0  , ...
    'saveoutput'     ,     true  , ...
    'parallel'       ,  parallelit);

fprintf('%s took %f seconds!\n', upper(engine), toc)

save('../Results/Joint_Mediation.mat','chains','diagnostics','info','stats')
