%OOS_MediationRT.m   Evaluated out-of-sample prediction
%                    for MediationRT model
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
%  10/14/17   Anna-Lena Schubert & Michael Nunez   Original code
%                   Adaptations for JAGS_OOS_noMediation.py output
%  01/22/18      Michael Nunez         Parameter changes for new samples
%  03/09/18     Michael Nunez          Conversion to releasable code
%  05/21/18     Michael Nunez           Change save location
%  11/27/18     Michael Nunez         Converted from OOS_Mediation.m


%% Model fit evaluation

addpath(genpath('..'));
load('../Data/ThirdData.mat'); % load IQ and ERP data
load('../Data/ThirdRT.mat');     % load RT data
data = ThirdSet;
N = size(data,1);

nSim = 150; %50 samples * 3 chains, see JAGS_OOS2_noMediation.py
noIQsamps = rdir('/home/michael/data10/michael/intel/jagsout/behavmodel_OOS2_noIQ_MediationRT_Nov_*.mat');
noERPsamps = rdir('/home/michael/data10/michael/intel/jagsout/behavmodel_OOS2_noERP_MediationRT_Nov_*.mat');
noRTsamps = rdir('/home/michael/data10/michael/intel/jagsout/behavmodel_OOS2_noAccRT_MediationRT_Nov_*.mat');

IQsim = [];
for j=1:length(noRTsamps)
    fprintf('Loading %s ...\n',noIQsamps(j).name);
    load(noIQsamps(j).name);
    IQpart = squeeze(permute(IQ, [5 1 2 3 4]));
    IQpart = reshape(IQpart, [nSim N 6]);
    IQsim = cat(1,IQsim,IQpart);
end

ERPsim = [];
for j=1:length(noRTsamps)
    fprintf('Loading %s ...\n',noERPsamps(j).name);
    load(noERPsamps(j).name);
    ERPpart = squeeze(permute(ERPdata, [5 1 2 3 4]));
    ERPpart = reshape(ERPpart, [nSim N 18]);
    ERPsim = cat(1,ERPsim,ERPpart);
end

ysim = [];
for j=1:length(noRTsamps)
    fprintf('Loading %s ...\n',noRTsamps(j).name);
    load(noRTsamps(j).name);
    ypart = squeeze(permute(y, [4 1 2 3]));
    ypart = reshape(ypart, [nSim,length(task)]);
    ysim = cat(1, ysim, ypart);
end

IQ = IQsim;
RTre = ysim;
ERPdata = ERPsim;
saveit = '../Results/Eval_MedRT_OOS.mat';
fprintf('Saving %s ...\n',saveit);
save(saveit,'IQ','RTre','ERPdata');

fprintf('Calculating statistics...\n');

%% Calculate summary statistics and R^2_prediction
%RT/Accuracy summary statistics: Mean, 25th percentile, Median, 75th percentile, Accuracy
summarystats = { @(x) (mean(x(x>0))); @(x) (prctile(x(x>0),25)); ...
    @(x) (median(x(x>0))); @(x) (prctile(x(x>0),75)); ...
    @(x) (sum(x>0)/length(x))};
r2pred = @(pred,realval) (1 - ((sum((pred-realval).^2)/(length(pred)-1))/var(realval))); %Anonymous function works for vectors 'pred' and 'realval' of equal length
nanr2pred = @(pred,realval) (1 - ((nansum((pred-realval).^2)/(sum(isfinite(pred))-1))/nanvar(realval))); %Anonymous function works for vectors 'pred' and 'realval' of equal length


% subjects from 1:N

subjects = unique(person);
P = length(subjects);
IDs = [];

for s = 1:P
    x = sum(person==subjects(s));
    ID = ones(x,1)*s;
    IDs = [IDs; ID];
end

person = IDs;

pred_medians = zeros(N,14);
real_medians = zeros(N,14);
pred_25th = zeros(N,14);
real_25th = zeros(N,14);
pred_75th = zeros(N,14);
real_75th = zeros(N,14);
pred_accuracies = zeros(N,14);
real_accuracies = zeros(N,14);
r2pred_medians = zeros(1,14);
r2pred_25th = zeros(1,14);
r2pred_75th = zeros(1,14);
r2pred_accuracies = zeros(1,14);
r2pred_medians_withnans = zeros(1,14);
r2pred_25th_withnans = zeros(1,14);
r2pred_75th_withnans = zeros(1,14);
r2pred_accuracies_withnans = zeros(1,14);


for t = 1:14,
    for s = 1:N,
        pred_medians(s,t) = summarystats{3}(RTre(person == s & task==t));
        real_medians(s,t) = summarystats{3}(y(person == s & task==t));
        pred_25th(s,t) = summarystats{2}(RTre(person == s & task==t));
        real_25th(s,t) = summarystats{2}(y(person == s & task==t));
        pred_75th(s,t) = summarystats{4}(RTre(person == s & task==t));
        real_75th(s,t) = summarystats{4}(y(person == s & task==t));
        pred_accuracies(s,t) = summarystats{5}(RTre(person == s & task==t));
        real_accuracies(s,t) = summarystats{5}(y(person == s & task==t));
        if sum(person == s & task==t) == 0,
            fprintf('Note that subject %d has no data for task %d \n',s,t);
        end 
    end
    r2pred_medians(t) = nanr2pred(pred_medians(:,t),real_medians(:,t));
    r2pred_25th(t) = nanr2pred(pred_25th(:,t),real_25th(:,t));
    r2pred_75th(t) = nanr2pred(pred_75th(:,t),real_75th(:,t));
    r2pred_accuracies(t) = nanr2pred(pred_accuracies(:,t),real_accuracies(:,t));
    r2pred_medians_withnans(t) = r2pred(pred_medians(:,t),real_medians(:,t));
    r2pred_25th_withnans(t) = r2pred(pred_25th(:,t),real_25th(:,t));
    r2pred_75th_withnans(t) = r2pred(pred_75th(:,t),real_75th(:,t));
    r2pred_accuracies_withnans(t) = r2pred(pred_accuracies(:,t),real_accuracies(:,t));
end

%% calculate mean R²

% z_r2_medians = atanh(r2pred_medians);
% mean_r2_median = mean(z_r2_medians);
% mean_r2_median = tanh(mean_r2_median);
mean_r2_median = mean(r2pred_medians);
fprintf('%.2f %% mean variance in RT medians described by out-of-sample prediction\n',mean_r2_median*100);
% mean R² = -2.3441

% z_r2_25th = atanh(r2pred_25th);
% mean_r2_25th = mean(z_r2_25th);
% mean_r2_25th = tanh(mean_r2_25th);
mean_r2_25th = mean(r2pred_25th);
fprintf('%.2f %% mean variance in RT 25th percentiles described by out-of-sample prediction\n',mean_r2_25th*100);
% mean R² =  - .3649

% z_r2_75th = atanh(r2pred_75th);
% mean_r2_75th = mean(z_r2_75th);
% mean_r2_75th = tanh(mean_r2_75th);
mean_r2_75th = mean(r2pred_75th);
fprintf('%.2f %% mean variance in RT 75th percentiles described by out-of-sample prediction\n',mean_r2_75th*100);
% mean R² = -1.1153

% z_r2_accuracies = atanh(r2pred_accuracies);
% mean_r2_accuracies = mean(z_r2_accuracies);
% mean_r2_accuracies = tanh(mean_r2_accuracies);
mean_r2_accuracies = mean(r2pred_accuracies(isfinite(r2pred_accuracies)));
fprintf('%.2f %% mean variance in accuracies described by out-of-sample prediction\n',mean_r2_accuracies*100);
% mean R² = -1.5790

%% Evaluate R² of IQ data

IQ_pred = squeeze(mean(IQ,1));
IQ_obs = data(:,2:7);

for t = 1:6
    % nonnanIQ = isfinite(IQ_obs(:,t));
 %    r2pred_IQ(t) = r2pred(IQ_pred(nonnanIQ,t),IQ_obs(nonnanIQ,t));
 r2pred_IQ(t) = nanr2pred(IQ_pred(:,t),IQ_obs(:,t));
end

%% calculate mean R²
% z_r2_IQ = atanh(r2pred_IQ);
% mean_r2_IQ = mean(z_r2_IQ);
% mean_r2_IQ = tanh(mean_r2_IQ);
mean_r2_IQ = mean(r2pred_IQ);
fprintf('%.2f %% mean variance in IQs described by out-of-sample prediction\n',mean_r2_IQ*100);
% mean R² = .2136

mean_IQ_pred = nanmean(IQ_pred,2);
mean_IQ_obs = nanmean(IQ_obs,2);


%% Evaluate R² of ERP data

ERP_pred = squeeze(mean(ERPdata,1));
ERP_obs = data(:,8:25);

for t = 1:18
    % nonnanERP = isfinite(ERP_obs(:,t));
 %    r2pred_ERP(t) = r2pred(ERP_pred(nonnanERP,t),ERP_obs(nonnanERP,t));
 r2pred_ERP(t) = nanr2pred(ERP_pred(:,t),ERP_obs(:,t));
end

% calculate mean R²
% z_r2_ERP = atanh(r2pred_ERP);
% mean_r2_ERP = mean(z_r2_ERP);
% mean_r2_ERP = tanh(mean_r2_ERP);
mean_r2_ERP = mean(r2pred_ERP);
fprintf('%.2f %% mean variance in ERPs described by out-of-sample prediction\n',mean_r2_ERP*100);
% mean R² = .1034