%ParameterStandardization.m     Standardizes regression weights for both
%                               models

% Copyright (C) 2018 Anna-Lena Schubert <anna-lena.schubert@psychologie.uni-heidelberg.de>
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
%  06/27/18    Anna-Lena Schubert          Original code
%  06/27/18        Michael Nunez           Load local data
%  07/17/18    Anna-Lena Schubert          Calculates intervals based on
%                                          whole posterior distribution

% General comments: Note that the variance of latent variables (unless
% fixed to 1) is determined both by factor loadings onto this variable and
% by residual variances. Hence,
%       Var(LV) = Var(b*higher-order LV) + Var(Residual of LV)
%               = b²*Var(higher-order LV) + Var(Residual of LV)
%       It follows that,
%       beta = b*sqrt(Var(higher-order LV)/(sqrt(b²*Var(higher-order LV) + Var(Residual of LV))

%% Simple regression model

% load data and extract relevant parameter samples from chains
load('../Results/Joint_noMediation.mat')
beta = chains.beta(:);
IQpsi = chains.IQpsi(:);
ERPpsi_1 = chains.ERPpsi_1(:);


% Regression of IQ test scores on ERP latencies
betaStandardized = prctile(beta.*(sqrt(ERPpsi_1)./sqrt((beta.^2).*ERPpsi_1 + IQpsi)), [2.5 50 97.5]);

% Path coefficients of ERP SEM
lambda_P2 = chains.ERPlambda_19(:);
lambda_N2 = chains.ERPlambda_20(:);
lambda_P3 = chains.ERPlambda_21(:);
psi_RP2 = chains.ERPpsi_2(:);
psi_RN2 = chains.ERPpsi_3(:);
psi_RP3 = chains.ERPpsi_4(:);
psi_ERP = chains.ERPpsi_1(:);
psi_P2 = psi_ERP + psi_RP2;
psi_N2 = (lambda_N2.^2).*psi_ERP + psi_RN2;
psi_P3 = (lambda_P3.^2).*psi_ERP + psi_RP3;
lambda_P2_Stand = lambda_P2.*(sqrt(psi_ERP)./sqrt(psi_P2));
lambda_N2_Stand = lambda_N2.*(sqrt(psi_ERP)./sqrt(psi_N2));
lambda_P3_Stand = lambda_P3.*(sqrt(psi_ERP)./sqrt(psi_P3));
median_lambda_ERP = [median(lambda_P2_Stand) median(lambda_N2_Stand) median(lambda_P3_Stand)];

lambda(1,1) = median(chains.ERPlambda_1(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_1(:).^2.*psi_P2 + chains.ERPtheta_1(:))));
lambda(2,1) = median(chains.ERPlambda_2(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_2(:).^2.*psi_P2 + chains.ERPtheta_2(:))));
lambda(3,1) = median(chains.ERPlambda_3(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_3(:).^2.*psi_P2 + chains.ERPtheta_3(:))));
lambda(4,1) = median(chains.ERPlambda_4(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_4(:).^2.*psi_P2 + chains.ERPtheta_4(:))));
lambda(5,1) = median(chains.ERPlambda_5(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_5(:).^2.*psi_P2 + chains.ERPtheta_5(:))));
lambda(6,1) = median(chains.ERPlambda_6(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_6(:).^2.*psi_P2 + chains.ERPtheta_6(:))));

lambda(7,1) = median(chains.ERPlambda_7(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_7(:).^2.*psi_N2 + chains.ERPtheta_7(:))));
lambda(8,1) = median(chains.ERPlambda_8(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_8(:).^2.*psi_N2 + chains.ERPtheta_8(:))));
lambda(9,1) = median(chains.ERPlambda_9(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_9(:).^2.*psi_N2 + chains.ERPtheta_9(:))));
lambda(10,1) = median(chains.ERPlambda_10(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_10(:).^2.*psi_N2 + chains.ERPtheta_10(:))));
lambda(11,1) = median(chains.ERPlambda_11(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_11(:).^2.*psi_N2 + chains.ERPtheta_11(:))));
lambda(12,1) = median(chains.ERPlambda_12(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_12(:).^2.*psi_N2 + chains.ERPtheta_12(:))));

lambda(13,1) = median(chains.ERPlambda_13(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_13(:).^2.*psi_P3 + chains.ERPtheta_13(:))));
lambda(14,1) = median(chains.ERPlambda_14(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_14(:).^2.*psi_P3 + chains.ERPtheta_14(:))));
lambda(15,1) = median(chains.ERPlambda_15(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_15(:).^2.*psi_P3 + chains.ERPtheta_15(:))));
lambda(16,1) = median(chains.ERPlambda_16(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_16(:).^2.*psi_P3 + chains.ERPtheta_16(:))));
lambda(17,1) = median(chains.ERPlambda_17(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_17(:).^2.*psi_P3 + chains.ERPtheta_17(:))));
lambda(18,1) = median(chains.ERPlambda_18(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_18(:).^2.*psi_P3 + chains.ERPtheta_18(:))));

% Path coefficients of Drift rate SEM

lambda_CR = chains.vlambda_15(:);
lambda_RM = chains.vlambda_16(:);
lambda_LM = chains.vlambda_17(:);
psi_RCR = chains.vpsi_2(:);
psi_RRM = chains.vpsi_3(:);
psi_RLM = chains.vpsi_4(:);
psi_v = chains.vpsi_1(:);
psi_CR = lambda_CR.^2.*psi_v + psi_RCR;
psi_RM = lambda_RM.^2.*psi_v + psi_RRM;
psi_LM = lambda_LM.^2.*psi_v + psi_RLM;
lambda_CR_Stand = lambda_CR.*(sqrt(psi_v)./sqrt(psi_CR));
lambda_RM_Stand = lambda_RM.*(sqrt(psi_v)./sqrt(psi_RM));
lambda_LM_Stand = lambda_LM.*(sqrt(psi_v)./sqrt(psi_LM));
median_lambda_v = [median(lambda_CR_Stand) median(lambda_RM_Stand) median(lambda_LM_Stand)];


lambda(1,2) = median(chains.vlambda_1(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_1(:).^2.*psi_CR +chains.vtheta_1(:))));
lambda(2,2) = median(chains.vlambda_2(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_2(:).^2.*psi_CR +chains.vtheta_2(:))));
lambda(3,2) = median(chains.vlambda_8(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_8(:).^2.*psi_CR +chains.vtheta_8(:))));
lambda(4,2) = median(chains.vlambda_9(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_9(:).^2.*psi_CR +chains.vtheta_9(:))));

lambda(5,2) = median(chains.vlambda_3(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_3(:).^2.*psi_RM +chains.vtheta_3(:))));
lambda(6,2) = median(chains.vlambda_4(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_4(:).^2.*psi_RM +chains.vtheta_4(:))));
lambda(7,2) = median(chains.vlambda_5(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_5(:).^2.*psi_RM +chains.vtheta_5(:))));
lambda(8,2) = median(chains.vlambda_10(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_10(:).^2.*psi_RM +chains.vtheta_10(:))));
lambda(9,2) = median(chains.vlambda_11(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_11(:).^2.*psi_RM +chains.vtheta_11(:))));
lambda(10,2) = median(chains.vlambda_12(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_12(:).^2.*psi_RM +chains.vtheta_12(:))));

lambda(11,2) = median(chains.vlambda_6(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_6(:).^2.*psi_LM +chains.vtheta_6(:))));
lambda(12,2) = median(chains.vlambda_7(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_7(:).^2.*psi_LM +chains.vtheta_7(:))));
lambda(13,2) = median(chains.vlambda_13(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_13(:).^2.*psi_LM +chains.vtheta_13(:))));
lambda(14,2) = median(chains.vlambda_14(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_14(:).^2.*psi_LM +chains.vtheta_14(:))));

% Path coefficients of IQ test scores SEM

psi_g = chains.beta(:).^2.*chains.ERPpsi_1(:) + chains.IQpsi(:);

lambda(1,3) = median(chains.IQlambda_1(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_1(:).^2.*psi_g + chains.IQtheta_1(:))));
lambda(2,3) = median(chains.IQlambda_2(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_2(:).^2.*psi_g + chains.IQtheta_2(:))));
lambda(3,3) = median(chains.IQlambda_3(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_3(:).^2.*psi_g + chains.IQtheta_3(:))));
lambda(4,3) = median(chains.IQlambda_4(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_4(:).^2.*psi_g + chains.IQtheta_4(:))));
lambda(5,3) = median(chains.IQlambda_5(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_5(:).^2.*psi_g + chains.IQtheta_5(:))));
lambda(6,3) = median(chains.IQlambda_6(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_6(:).^2.*psi_g + chains.IQtheta_6(:))));

%% Mediation model

% load data and extract relevant parameter samples from chains
load('../Results/Joint_Mediation.mat')
beta(:,1) = chains.beta_1(:);
beta(:,2) = chains.beta_2(:);
beta(:,3) = chains.beta_3(:);

IQpsi = chains.IQpsi(:);
ERPpsi_1 = chains.ERPpsi_1(:);
vpsi_1 = chains.vpsi_1(:);


% Regression of drift rates on ERP latencies
betaStandardized(1,:) = prctile(beta(:,1).*sqrt(ERPpsi_1)./sqrt((beta(:,1).^2.*ERPpsi_1 + vpsi_1)), [2.5 50 97.5]);

%Regression of IQ test scores on drift rates
betaStandardized(2,:) = prctile(beta(:,2).*(sqrt(((beta(:,1).^2).*ERPpsi_1 + vpsi_1))./ ...
    sqrt((beta(:,2).^2).*((beta(:,1).^2).*ERPpsi_1 + vpsi_1) + ...
    (beta(:,3).^2).*ERPpsi_1 + IQpsi + 2.*beta(:,1).*beta(:,2).*beta(:,3).*ERPpsi_1)), ...
    [2.5 50 97.5]);

% Regression of IQ test scores on ERP latencies
betaStandardized(3,:) = prctile(beta(:,3).*(sqrt(ERPpsi_1)./ ...
    sqrt(((beta(:,2).^2).*((beta(:,1).^2).*ERPpsi_1 + vpsi_1) + ...
    (beta(:,3).^2).*ERPpsi_1 + IQpsi + 2.*beta(:,1).*beta(:,2).*beta(:,3).*ERPpsi_1))), ...
    [2.5 50 97.5]);

% Indirect effect
betaStandardized(4,:) = prctile((beta(:,1).*sqrt(ERPpsi_1)./sqrt((beta(:,1).^2.*ERPpsi_1 + vpsi_1))).* ...
    (beta(:,2).*(sqrt(((beta(:,1).^2).*ERPpsi_1 + vpsi_1))./ ...
    sqrt((beta(:,2).^2).*((beta(:,1).^2).*ERPpsi_1 + vpsi_1) + ...
    (beta(:,3).^2).*ERPpsi_1 + IQpsi + 2.*beta(:,1).*beta(:,2).*beta(:,3).*ERPpsi_1))), ...
    [2.5 50 97.5]);

% Path coefficients of ERP SEM
lambda_P2 = chains.ERPlambda_19(:);
lambda_N2 = chains.ERPlambda_20(:);
lambda_P3 = chains.ERPlambda_21(:);
psi_RP2 = chains.ERPpsi_2(:);
psi_RN2 = chains.ERPpsi_3(:);
psi_RP3 = chains.ERPpsi_4(:);
psi_ERP = chains.ERPpsi_1(:);
psi_P2 = psi_ERP + psi_RP2;
psi_N2 = (lambda_N2.^2).*psi_ERP + psi_RN2;
psi_P3 = (lambda_P3.^2).*psi_ERP + psi_RP3;
lambda_P2_Stand = lambda_P2.*(sqrt(psi_ERP)./sqrt(psi_P2));
lambda_N2_Stand = lambda_N2.*(sqrt(psi_ERP)./sqrt(psi_N2));
lambda_P3_Stand = lambda_P3.*(sqrt(psi_ERP)./sqrt(psi_P3));
median_lambda_ERP = [median(lambda_P2_Stand) median(lambda_N2_Stand) median(lambda_P3_Stand)];

lambda(1,1) = median(chains.ERPlambda_1(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_1(:).^2.*psi_P2 + chains.ERPtheta_1(:))));
lambda(2,1) = median(chains.ERPlambda_2(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_2(:).^2.*psi_P2 + chains.ERPtheta_2(:))));
lambda(3,1) = median(chains.ERPlambda_3(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_3(:).^2.*psi_P2 + chains.ERPtheta_3(:))));
lambda(4,1) = median(chains.ERPlambda_4(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_4(:).^2.*psi_P2 + chains.ERPtheta_4(:))));
lambda(5,1) = median(chains.ERPlambda_5(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_5(:).^2.*psi_P2 + chains.ERPtheta_5(:))));
lambda(6,1) = median(chains.ERPlambda_6(:).*(sqrt(psi_P2)./ ...
    sqrt(chains.ERPlambda_6(:).^2.*psi_P2 + chains.ERPtheta_6(:))));

lambda(7,1) = median(chains.ERPlambda_7(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_7(:).^2.*psi_N2 + chains.ERPtheta_7(:))));
lambda(8,1) = median(chains.ERPlambda_8(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_8(:).^2.*psi_N2 + chains.ERPtheta_8(:))));
lambda(9,1) = median(chains.ERPlambda_9(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_9(:).^2.*psi_N2 + chains.ERPtheta_9(:))));
lambda(10,1) = median(chains.ERPlambda_10(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_10(:).^2.*psi_N2 + chains.ERPtheta_10(:))));
lambda(11,1) = median(chains.ERPlambda_11(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_11(:).^2.*psi_N2 + chains.ERPtheta_11(:))));
lambda(12,1) = median(chains.ERPlambda_12(:).*(sqrt(psi_N2)./ ...
    sqrt(chains.ERPlambda_12(:).^2.*psi_N2 + chains.ERPtheta_12(:))));

lambda(13,1) = median(chains.ERPlambda_13(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_13(:).^2.*psi_P3 + chains.ERPtheta_13(:))));
lambda(14,1) = median(chains.ERPlambda_14(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_14(:).^2.*psi_P3 + chains.ERPtheta_14(:))));
lambda(15,1) = median(chains.ERPlambda_15(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_15(:).^2.*psi_P3 + chains.ERPtheta_15(:))));
lambda(16,1) = median(chains.ERPlambda_16(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_16(:).^2.*psi_P3 + chains.ERPtheta_16(:))));
lambda(17,1) = median(chains.ERPlambda_17(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_17(:).^2.*psi_P3 + chains.ERPtheta_17(:))));
lambda(18,1) = median(chains.ERPlambda_18(:).*(sqrt(psi_P3)./ ...
    sqrt(chains.ERPlambda_18(:).^2.*psi_P3 + chains.ERPtheta_18(:))));

% Path coefficients of Drift rate SEM

psi_v = chains.beta_1(:).^2.*chains.ERPpsi_1(:) + chains.vpsi_1(:);
lambda_CR = chains.vlambda_15(:);
lambda_RM = chains.vlambda_16(:);
lambda_LM = chains.vlambda_17(:);
psi_RCR = chains.vpsi_2(:);
psi_RRM = chains.vpsi_3(:);
psi_RLM = chains.vpsi_4(:);
psi_CR = lambda_CR.^2.*psi_v + psi_RCR;
psi_RM = lambda_RM.^2.*psi_v + psi_RRM;
psi_LM = lambda_LM.^2.*psi_v + psi_RLM;
lambda_CR_Stand = lambda_CR.*(sqrt(psi_v)./sqrt(psi_CR));
lambda_RM_Stand = lambda_RM.*(sqrt(psi_v)./sqrt(psi_RM));
lambda_LM_Stand = lambda_LM.*(sqrt(psi_v)./sqrt(psi_LM));
median_lambda_v = [median(lambda_CR_Stand) median(lambda_RM_Stand) median(lambda_LM_Stand)];

lambda(1,2) = median(chains.vlambda_1(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_1(:).^2.*psi_CR +chains.vtheta_1(:))));
lambda(2,2) = median(chains.vlambda_2(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_2(:).^2.*psi_CR +chains.vtheta_2(:))));
lambda(3,2) = median(chains.vlambda_8(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_8(:).^2.*psi_CR +chains.vtheta_8(:))));
lambda(4,2) = median(chains.vlambda_9(:).*(sqrt(psi_CR)./ ...
    sqrt(chains.vlambda_9(:).^2.*psi_CR +chains.vtheta_9(:))));

lambda(5,2) = median(chains.vlambda_3(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_3(:).^2.*psi_RM +chains.vtheta_3(:))));
lambda(6,2) = median(chains.vlambda_4(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_4(:).^2.*psi_RM +chains.vtheta_4(:))));
lambda(7,2) = median(chains.vlambda_5(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_5(:).^2.*psi_RM +chains.vtheta_5(:))));
lambda(8,2) = median(chains.vlambda_10(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_10(:).^2.*psi_RM +chains.vtheta_10(:))));
lambda(9,2) = median(chains.vlambda_11(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_11(:).^2.*psi_RM +chains.vtheta_11(:))));
lambda(10,2) = median(chains.vlambda_12(:).*(sqrt(psi_RM)./ ...
    sqrt(chains.vlambda_12(:).^2.*psi_RM +chains.vtheta_12(:))));

lambda(11,2) = median(chains.vlambda_6(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_6(:).^2.*psi_LM +chains.vtheta_6(:))));
lambda(12,2) = median(chains.vlambda_7(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_7(:).^2.*psi_LM +chains.vtheta_7(:))));
lambda(13,2) = median(chains.vlambda_13(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_13(:).^2.*psi_LM +chains.vtheta_13(:))));
lambda(14,2) = median(chains.vlambda_14(:).*(sqrt(psi_LM)./ ...
    sqrt(chains.vlambda_14(:).^2.*psi_LM +chains.vtheta_14(:))));

% Path coefficients of IQ test scores SEM

psi_g = chains.beta_2(:).^2.*(chains.beta_1(:).^2.*chains.ERPpsi_1(:) + chains.vpsi_1(:)) + ...
    chains.beta_3(:).^2.*chains.ERPpsi_1(:) + chains.IQpsi(:) + ...
    2.*chains.beta_1(:).*chains.beta_2(:).*chains.beta_3(:).*chains.ERPpsi_1(:);

lambda(1,3) = median(chains.IQlambda_1(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_1(:).^2.*psi_g + chains.IQtheta_1(:))));
lambda(2,3) = median(chains.IQlambda_2(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_2(:).^2.*psi_g + chains.IQtheta_2(:))));
lambda(3,3) = median(chains.IQlambda_3(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_3(:).^2.*psi_g + chains.IQtheta_3(:))));
lambda(4,3) = median(chains.IQlambda_4(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_4(:).^2.*psi_g + chains.IQtheta_4(:))));
lambda(5,3) = median(chains.IQlambda_5(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_5(:).^2.*psi_g + chains.IQtheta_5(:))));
lambda(6,3) = median(chains.IQlambda_6(:).*(sqrt(psi_g)./ ...
    sqrt(chains.IQlambda_6(:).^2.*psi_g + chains.IQtheta_6(:))));

%% P3 model

beta(:,1) = chains.beta_1(:);
beta(:,2) = chains.beta_2(:);
beta(:,3) = chains.beta_3(:);
beta(:,4) = chains.beta_4(:);
beta(:,5) = chains.beta_5(:);


beta_CI_1 = prctile(beta(:,1),[2.5 50 97.5])
beta_CI_2 = prctile(beta(:,2),[2.5 50 97.5])
beta_CI_3 = prctile(beta(:,3),[2.5 50 97.5])
beta_CI_4 = prctile(beta(:,4),[2.5 50 97.5])
beta_CI_5 = prctile(beta(:,5),[2.5 50 97.5])

%% Plotting

load('../Results/Joint_Mediation.mat')
beta(:,1) = chains.beta_1(:);
beta(:,2) = chains.beta_2(:);
beta(:,3) = chains.beta_3(:);


IQpsi = chains.IQpsi(:);
ERPpsi_1 = chains.ERPpsi_1(:);
vpsi_1 = chains.vpsi_1(:);


betaStandardized(1,:) = beta(:,1).*(sqrt(ERPpsi_1)./sqrt((beta(:,1).^2).*ERPpsi_1 + vpsi_1));
betaStandardized(2,:) = beta(:,2).*(sqrt(((beta(:,1).^2).*ERPpsi_1 + vpsi_1))./ ...
    sqrt((beta(:,2).^2).*((beta(:,1).^2).*ERPpsi_1 + vpsi_1) + ...
    (beta(:,3).^2).*ERPpsi_1 + IQpsi + 2*beta(:,1).*beta(:,2).*beta(:,3).*ERPpsi_1));
betaStandardized(3,:) = beta(:,3).*(sqrt(ERPpsi_1)./ ...
    sqrt((beta(:,2).^2).*((beta(:,1).^2).*ERPpsi_1 + vpsi_1) + ...
    (beta(:,3).^2).*ERPpsi_1 + IQpsi + 2*beta(:,1).*beta(:,2).*beta(:,3).*ERPpsi_1));
betaStandardized(4,:) = betaStandardized(1,:).*betaStandardized(2,:);

csvwrite('../Results/RegressionWeights.csv', betaStandardized')
