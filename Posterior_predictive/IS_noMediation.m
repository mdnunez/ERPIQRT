%IS_noMediation.m   Evaluated in-sample prediction
%                    for noMediation model

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
%  01/23/18     Anna-Lena Schubert & Michael Nunez        Converted from OOS_noMediation.m
%                                and ModelEvaluation.m
%  02/01/18     Michael Nunez          Conversion to releasable code
%  03/02/18     Michael Nunez         Evaluate joint data
%  03/06/18     Michael Nunez     Fixes for missing data

%%%UNFINISHED%%%

%% Model fit evaluation
addpath(genpath('..'));
chainLength = 30000; %chain length
firstset = load('../Data/data.mat');        % load IQ and ERP data
secondset = load('../Data/SecondData.mat');
firstsetRT = load('../Data/RTdata.mat');      % load RT data
secondsetRT = load('../Data/SecondRT.mat');
load('~/data10/michael/intel/Results/Joint_noMediation.mat');

jagsout = readjagsout(stats,diagnostics);
fprintf('The maximum Gelman-Rubin statistic is %.3f \n',max(jagsout.Rhat));
fprintf('The minimum number of effective samples is %.3f \n',min(jagsout.Neff));


bothdata = [firstset.data ; secondset.SecondSet];
N = numel(bothdata(:,1));   % number of participants

person = [firstsetRT.person ; secondsetRT.person];
task = [firstsetRT.task ; secondsetRT.task];
y = [firstsetRT.y ; secondsetRT.y];

IQ_mu = zeros(chainLength,N,6);
ERP_mu = zeros(chainLength,N,18);
IQ_theta = zeros(chainLength,6);
ERP_theta = zeros(chainLength,18);
v = zeros(chainLength,14,N);
a = zeros(chainLength,14,N);
ter = zeros(chainLength,14,N);

if exist('../../final_results/Eval_noMed_IS.mat') ~= 2
    rng(13);
    for i = 1:N
        for k = 1:6
            IQ_mu(:,i,k) = chains.(sprintf('IQmu_%d_%d',k,i))(:);
        end
        
        for k = 1:18
            ERP_mu(:,i,k) = chains.(sprintf('ERPmu_%d_%d',i,k))(:);
        end
        
        for k = 1:14
            v(:,k,i) = chains.(sprintf('v_%d_%d',k,i))(:);
            a(:,k,i) = chains.(sprintf('a_%d_%d',k,i))(:);
            ter(:,k,i) = chains.(sprintf('ter_%d_%d',k,i))(:);
        end
        
    end

    for k = 1:6
        IQ_theta(:,k) = chains.(sprintf('IQtheta_%d',k))(:);
    end

    for k = 1:18
        ERP_theta(:,k) = chains.(sprintf('ERPtheta_%d',k))(:);
    end


    nSim = 1000;
    selectedChains = randsample(chainLength,nSim);
    IQ = zeros(nSim,N,6);
    ERPdata = zeros(nSim,N,18);
    RTdata = zeros(nSim,length(task));
    ACCdata = zeros(nSim,length(task));

    subjects = unique(person);
    P = length(subjects);
    IDs = [];

    for s = 1:P
        x = sum(person==subjects(s));
        ID = ones(x,1)*s;
        IDs = [IDs; ID];
    end

    person = IDs; X=[];

    for c = 1:nSim
        fprintf('Simulation %d\n',c);
        for i = 1:N
            for j = 1:6
                IQ(c,i,j) = normrnd(IQ_mu(selectedChains(c),i,j), ...
                    sqrt(IQ_theta(selectedChains(c),j))); %Use standard deviation
            end
            
            for j = 1:18
                ERPdata(c,i,j) = normrnd(ERP_mu(selectedChains(c),i,j), ...
                    sqrt(ERP_theta(selectedChains(c),j))); %Use standard deviation
            end
        end
        
        for i = 1:length(task)
            
            if v(selectedChains(c),task(i),person(i))> 5
                v_ID = 0.5;
            else
                v_ID = v(selectedChains(c),task(i),person(i))/10;
            end
            
            [RT,ACC] = simuldiff([a(selectedChains(c),task(i),person(i))/10, ...
                ter(selectedChains(c),task(i),person(i)), ...
                0, ...
                0.5*a(selectedChains(c),task(i),person(i))/10, ...
                0, ...
                0, ...
                v_ID], ...
                1);
            
            RTdata(c,i) = RT;
            ACCdata(c,i) = ACC;
        end
    end

    for i = 1:1000
        
        for j =1:length(y)
            if ACCdata(i,j) == 0
                RTre(i,j) = -1*RTdata(i,j);
            else
                RTre(i,j) = RTdata(i,j);
            end
        end
    end
    RTre = RTre';
save('Eval_noMed_IS.mat','IQ','RTdata','ACCdata','RTre','ERPdata')
else
    fprintf('Loading found simulated data...\n');
    load('../../final_results/Eval_noMed_IS.mat');
end



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

% In addition, some participants didn't work on all tasks, which causes
% NaNs to appear in the summary below, how do we fix that?
% See person == 41 & task==13

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
fprintf('%.2f %% mean variance in RT medians described by in-sample prediction\n',mean_r2_median*100);
% mean R² = .90

% z_r2_25th = atanh(r2pred_25th);
% mean_r2_25th = mean(z_r2_25th);
% mean_r2_25th = tanh(mean_r2_25th);
mean_r2_25th = mean(r2pred_25th);
fprintf('%.2f %% mean variance in RT 25th percentiles described by in-sample prediction\n',mean_r2_25th*100);
% mean R² = .88

% z_r2_75th = atanh(r2pred_75th);
% mean_r2_75th = mean(z_r2_75th);
% mean_r2_75th = tanh(mean_r2_75th);
mean_r2_75th = mean(r2pred_75th);
fprintf('%.2f %% mean variance in RT 75th percentiles described by in-sample prediction\n',mean_r2_75th*100);
% mean R² = .83

% z_r2_accuracies = atanh(r2pred_accuracies);
% mean_r2_accuracies = mean(z_r2_accuracies);
% mean_r2_accuracies = tanh(mean_r2_accuracies);
mean_r2_accuracies = mean(r2pred_accuracies(isfinite(r2pred_accuracies)));
fprintf('%.2f %% mean variance in accuracies described by in-sample prediction\n',mean_r2_accuracies*100);
% mean R² = .24

%% Evaluate R² of IQ data

IQ_pred = squeeze(mean(IQ,1));
IQ_obs = bothdata(:,2:7);

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
fprintf('%.2f %% mean variance in IQs described by in-sample prediction\n',mean_r2_IQ*100);
% mean R² = .58

mean_IQ_pred = nanmean(IQ_pred,2);
mean_IQ_obs = nanmean(IQ_obs,2);


%% Evaluate R² of ERP data

ERP_pred = squeeze(mean(ERPdata,1));
ERP_obs = bothdata(:,8:25);

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
fprintf('%.2f %% mean variance in ERPs described by in-sample prediction\n',mean_r2_ERP*100);
% mean R² = .55

mean_ERP_pred = nanmean(ERP_pred,2);
mean_ERP_obs = nanmean(ERP_obs,2);

betastats = prctile(chains.beta(:),[2.5 50 97.5]);
fprintf('Effect (median posterior and 95%% credible interval) of latent neural processing on latent cognitive ability: %.2f, CI: [%.2f, %.2f]\n',betastats(2),betastats(1),betastats(3));