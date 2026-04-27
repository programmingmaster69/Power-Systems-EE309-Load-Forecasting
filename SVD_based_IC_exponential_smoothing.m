clc; clear; close all;
rng(1);

%% SETTINGS
trainFile = 'data/Training sheet.xlsx';
testFile  = 'data/Testing sheet.xlsx';
colIdx    = 3;   % Column C (actual load)

m1 = 48;
m2 = 336;
H  = 48;
k  = 31;

alphaGrid = [0.001 0.005 0.01];
deltaGrid = [0.10 0.30 0.50];
omegaGrid = [0.10 0.30 0.50];
phiGrid   = [0.00 0.50 0.90];

fitFraction = 0.80;

%% READ DATA
trainRaw = readtable(trainFile);
testRaw  = readtable(testFile);

trainRaw = trainRaw{:,colIdx};
testRaw  = testRaw{:,colIdx};

trainRaw = trainRaw(isfinite(trainRaw));
testRaw  = testRaw(isfinite(testRaw));

%% LOG TRANSFORM
trainLog = log(trainRaw);
testLog  = log(testRaw);

%% SPLIT
nTrain = numel(trainLog);
nFit   = floor((fitFraction*nTrain)/m2)*m2;

fitSeries = trainLog(1:nFit);
valSeries = trainLog(nFit+1:end);

%% SVD
[Vfit,~,p0fit] = buildWeeklySVD(fitSeries, m2, k);

%% PARAMETER TUNING (LOG DOMAIN)
bestScore = inf;

for alpha = alphaGrid
for delta = deltaGrid
for omega = omegaGrid
for phi = phiGrid

    params = struct('alpha',alpha,'delta',delta,'omega',omega,'phi',phi);

    [~,stateHistFit,errHistFit] = onlineFilter( ...
        fitSeries,Vfit,params,m1,m2,p0fit,1,0);

    stateEnd = stateHistFit(:,end);
    lastErr  = errHistFit(end);

    [valPredLog,~,~] = onlineFilter( ...
        valSeries,Vfit,params,m1,m2,stateEnd,nFit+1,lastErr);

    % LOG-domain MAPE
    score = mean(abs(valSeries - valPredLog) ./ max(abs(valSeries),1e-6)) * 100;

    if score < bestScore
        bestScore = score;
        bestParams = params;
    end

end
end
end
end

fprintf('Best params: alpha=%.5f delta=%.3f omega=%.3f phi=%.3f\n', ...
    bestParams.alpha,bestParams.delta,bestParams.omega,bestParams.phi);

%% FINAL MODEL
[Vfull,~,p0full] = buildWeeklySVD(trainLog,m2,k);

combinedLog = [trainLog; testLog];

[allPredLog,stateHistAll,errHistAll] = onlineFilter( ...
    combinedLog,Vfull,bestParams,m1,m2,p0full,1,0);

trainPredLog = allPredLog(1:nTrain);
testPredLog  = allPredLog(nTrain+1:end);

%% ACTUAL VS FORECAST (LOG SCALE)
figure;
plot(trainLog,'b'); hold on;
plot(trainPredLog,'r');
legend('Actual (log)','Forecast (log)');
title('Training Data (Log Scale)');

figure;
plot(testLog,'b'); hold on;
plot(testPredLog,'r');
legend('Actual (log)','Forecast (log)');
title('Testing Data (Log Scale)');

%% ROLLING FORECAST (LOG DOMAIN APE)
nTest = numel(testLog);
APE = NaN(H,nTest-H);

for o = 1:(nTest-H)

    origin = nTrain + o - 1;
    state = stateHistAll(:,origin);
    lastErr = errHistAll(origin);

    for h = 1:H
        idx = origin + h;
        weekIdx = mod(idx-1,m2)+1;

        if h==1
            e = lastErr;
        else
            e = 0;
        end

        yhatLog = state'*Vfull(weekIdx,:)' + bestParams.phi*e;
        ytrueLog = combinedLog(idx);

        % LOG-domain APE
        APE(h,o) = abs(ytrueLog - yhatLog) / max(abs(ytrueLog),1e-6) * 100;
    end
end

MAPE = mean(APE,2,'omitnan');

%% =====================================================
% IEEE STYLE PLOTS
%% =====================================================

% Common styling
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');

%% =====================================================
% FIGURE 1: TRAINING (Actual vs Forecast)
%% =====================================================
figure('Color','w','Position',[100 100 700 400]);

plot(trainLog,'b','LineWidth',1.6); hold on;
plot(trainPredLog,'r','LineWidth',1.6);

grid on; box on;

xlabel('Time Index','FontSize',12,'FontWeight','bold');
ylabel('Load (Log Scale)','FontSize',12,'FontWeight','bold');

legend({'Actual Load','Forecasted Load'}, ...
    'Location','best','FontSize',10);

title('Training Data: Actual vs Forecast (Log Domain)', ...
    'FontSize',12,'FontWeight','bold');

set(gca,'FontSize',11,'LineWidth',1.2);

exportgraphics(gcf,'Fig_Train_Actual_vs_Forecast.png','Resolution',600);

%% =====================================================
% FIGURE 2: TESTING (Actual vs Forecast)
%% =====================================================
figure('Color','w','Position',[120 120 700 400]);

plot(testLog,'b','LineWidth',1.6); hold on;
plot(testPredLog,'r','LineWidth',1.6);

grid on; box on;

xlabel('Time Index','FontSize',12,'FontWeight','bold');
ylabel('Load (Log Scale)','FontSize',12,'FontWeight','bold');

legend({'Actual Load','Forecasted Load'}, ...
    'Location','best','FontSize',10);

title('Testing Data: Actual vs Forecast (Log Domain)', ...
    'FontSize',12,'FontWeight','bold');

set(gca,'FontSize',11,'LineWidth',1.2);

exportgraphics(gcf,'Fig_Test_Actual_vs_Forecast.png','Resolution',600);

%% =====================================================
% FIGURE 3: MAPE vs HORIZON
%% =====================================================
figure('Color','w','Position',[140 140 700 400]);

plot((1:H)/2,MAPE,'k-o',...
    'LineWidth',1.8,...
    'MarkerSize',4,...
    'MarkerFaceColor','k');

grid on; box on;

xlabel('Forecast Horizon (Hours)',...
    'FontSize',12,'FontWeight','bold');

ylabel('MAPE (\%) [Log Domain]',...
    'FontSize',12,'FontWeight','bold');

title('MAPE vs Forecast Horizon', ...
    'FontSize',12,'FontWeight','bold');

set(gca,'FontSize',11,'LineWidth',1.2);

exportgraphics(gcf,'Fig_MAPE_vs_Horizon.png','Resolution',600);

%% =====================================================
% FIGURE 4: APE DISTRIBUTION (BOXPLOT)
%% =====================================================
figure('Color','w','Position',[160 160 700 400]);

boxplot(APE','Symbol','k.','Whisker',1.5);

grid on; box on;

xlabel('Forecast Horizon (Half-Hours)',...
    'FontSize',12,'FontWeight','bold');

ylabel('APE (\%) [Log Domain]',...
    'FontSize',12,'FontWeight','bold');

title('APE Distribution vs Forecast Horizon', ...
    'FontSize',12,'FontWeight','bold');

set(gca,'FontSize',11,'LineWidth',1.2);

exportgraphics(gcf,'Fig_APE_Boxplot.png','Resolution',600);

%% =====================================================
% EXPORT: MAPE vs HORIZON TABLE
%% =====================================================

% Horizon in hours (since 48 half-hours = 24 hours)
horizon_hours = (1:H)'/2;

% Create table
MAPE_Table = table(horizon_hours, MAPE, ...
    'VariableNames', {'Horizon_Hours','MAPE_percent'});

% Display in MATLAB
disp('MAPE vs Horizon Table:');
disp(MAPE_Table);

% Export to CSV
writetable(MAPE_Table,'MAPE_vs_Horizon.csv');

% Export to Excel (better for paper)
writetable(MAPE_Table,'MAPE_vs_Horizon.xlsx');

disp('Saved Files:');
disp('1. MAPE_vs_Horizon.csv');
disp('2. MAPE_vs_Horizon.xlsx');

function [V,P,p0] = buildWeeklySVD(series,m2,k)

series = series(:);
nWeeks = floor(numel(series)/m2);

Y = reshape(series(1:nWeeks*m2),m2,nWeeks)';

[U,S,Vfull] = svd(Y,'econ');

k = min([k,size(Vfull,2)]);
V = Vfull(:,1:k);
P = U(:,1:k)*S(1:k,1:k);

p0 = P(1,:)';
end

function [yhat,stateHist,errHist] = onlineFilter(series,V,params,m1,m2,p0,startIdx,lastErr0)

n = numel(series);
k = size(V,2);

yhat = NaN(n,1);
stateHist = NaN(k,n);
errHist = NaN(n,1);

state = p0;
lastErr = lastErr0;

for t = 1:n
    globalIdx = startIdx + t - 1;
    weekIdx = mod(globalIdx-1,m2)+1;

    yhat(t) = state'*V(weekIdx,:)' + params.phi*lastErr;

    if t==1
        e = 0;
    else
        e = series(t) - yhat(t);
        state = updateState(state,V,weekIdx,e,params,m1,m2);
        lastErr = e;
    end

    stateHist(:,t) = state;
    errHist(t) = lastErr;
end
end

function newState = updateState(state,V,weekIdx,e,params,m1,m2)

k = length(state);
intradayIdx = mod(weekIdx-1,m1)+1;

basisSum = zeros(k,1);

for j = 1:7
    idx2 = intradayIdx + (j-1)*m1;
    if idx2 <= m2
        basisSum = basisSum + V(idx2,:)';
    end
end

updateVec = params.alpha*ones(k,1) + ...
            params.delta*basisSum + ...
            params.omega*V(weekIdx,:)';

newState = state + updateVec * e;
end

