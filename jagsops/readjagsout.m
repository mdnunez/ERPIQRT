function jagsout = readjagsout(stats,diagnostics)
%READJAGSOUT -Places JAGS output using trinity into manipulable format
%
%Usage: jagsout = readjagsout(stats,diagnostics)
%
%
%Inputs:
%  stats: Structure containing posterior means,stds,etc. (output of trinity)
%  diagnostics: Structure containing model diagnostics (output of trinity)
%
%Outputs:
% jagsout: Structure with matrix and vector fields

% Copyright (C) 2015 Michael D. Nunez <mdnunez1@uci.edu>
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

%% Record of Revisions
%   Date           Programmers               Description of change
%   ====        =================            =====================
%  04/28/15        Michael Nunez                  Original Code
%  11/30/15        Michael Nunez                Default inputs

%% Initial

if nargin < 1 || isempty(stats)
    stats = evalin('base','stats');
end

if nargin < 2 || isempty(diagnostics)
    diagnostics = evalin('base','diagnostics');
end
%% Code

jagsout.params = fieldnames(stats.mean);

statsfields = fieldnames(stats);
for f=1:length(statsfields)
    jagsout.(statsfields{f}) = cell2mat(struct2cell(stats.(statsfields{f})));
end


diagfields = fieldnames(diagnostics);
for f=1:length(diagfields)
    jagsout.(diagfields{f}) = cell2mat(struct2cell(diagnostics.(diagfields{f})));
end