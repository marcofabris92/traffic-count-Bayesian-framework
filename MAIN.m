clear
clc
close all

%% ==============================================================
% DATA LOADING
% ==============================================================

% NOTE:
% - Column 1: time stamps
% - Column 2: oracle signal (Video)
% - Columns 3:end: sensor signals (vehicle classes)

%Sensors = readtable('DatiTest.csv');        % Binary version
Sensors = readtable('DatiTest_ClasseVeic.csv'); % Multi-class version

Time_stamps = table2array(Sensors(:,1));
Video       = table2array(Sensors(:,2));     % Oracle (ground truth)
Sensors     = table2array(Sensors(:,3:end)); % Sensor measurements

numSensors = size(Sensors,2);

%% --------------------------------------------------------------
% Select a time window from the dataset
% --------------------------------------------------------------
% The dataset is divided into blocks of length T_0.
% kk selects which block to analyze.

T_0 = 2*3600;   % 2 hours expressed in seconds
kk  = 1;        % block index (0-based logic shifted by 1)

tm_x = 1 + kk*T_0;      % initial time index for oracle
tM_x = T_0 + kk*T_0;    % final time index for oracle

% For sensors, initially assume same window
tm_y = tm_x * ones(numSensors,1);
tM_y = tM_x * ones(numSensors,1);

%% ==============================================================
% DETRENDING (SYSTEMATIC LAG ESTIMATION)
% ==============================================================

% --------------------------------------------------------------
% Sanity check: verify overlapping intervals exist
% --------------------------------------------------------------

parfor n = 1:numSensors

    tm = max(tm_x, tm_y(n));
    tM = min(tM_x, tM_y(n));

    % If no overlap, systematic lag cannot be estimated
    if tM <= tm
        error('Bad data selection on Sensor %d\n', n);
    end
end

% --------------------------------------------------------------
% Estimate systematic lag mu(n) for each sensor
% --------------------------------------------------------------
% mu(n) aligns sensor n to the oracle using a
% penalized Hamming-distance criterion.

mu = zeros(numSensors,1);

parfor n = 1:numSensors
    mu(n) = estimate_lag_hh( ...
        Video, tm_x, tM_x, ...
        Sensors(:,n), tm_y(n), tM_y(n));
end

%% --------------------------------------------------------------
% Build detrended (aligned) signals
% --------------------------------------------------------------
% After estimating mu(n), we shift each sensor accordingly
% and compute the common overlapping interval.

tm_yD = tm_y - mu;
tM_yD = tM_y - mu;

% Enforce valid bounds
parfor n = 1:numSensors
    if tm_yD(n) < tm_y(n)
        tm_yD(n) = tm_y(n);
    end
    if tM_yD(n) > tM_y(n)
        tM_yD(n) = tM_y(n);
    end
end

% Global common time window across oracle and all sensors
tm = max([tm_x; tm_yD]);
tM = min([tM_x; tM_yD]);

T = tM - tm + 1;    % Effective aligned signal length

% Allocate aligned signals
x = zeros(T,1);               % Oracle (aligned)
y = zeros(T,numSensors);      % Sensors (aligned)

% Build aligned sequences
for t = tm:tM

    k = t - tm + 1;   % Local index

    x(k) = Video(t);

    for n = 1:numSensors
        y(k,n) = Sensors(t + mu(n), n);
    end
end

% Store estimated shifts
shifts = mu';
shifts

%% ==============================================================
% MODEL SELECTION OVER TEMPORAL WINDOW Δ
% ==============================================================

% Δ defines the maximum admissible delay between oracle and sensor
DeltaMax = 30;

criterion_score = zeros(DeltaMax, numSensors);
models = cell(DeltaMax,1);

for Delta = 0:DeltaMax

    logL  = zeros(numSensors,1);
    model = struct();

    %% ----------------------------------------------------------
    % Estimate θ_V (oracle activity probability)
    % ----------------------------------------------------------
    % θ_V = probability that at least one oracle event
    % occurs within ±Δ around time k.

    thetaV_hat = 0;

    for k = 1:T
        for d = -Delta:Delta

            kd = k + d;

            if kd >= 1 && kd <= T && chi(x(kd))
                thetaV_hat = thetaV_hat + 1;
                break;  % stop at first detection
            end
        end
    end

    thetaV_hat = thetaV_hat / T;

    %% ----------------------------------------------------------
    % Per-sensor parameter estimation
    % ----------------------------------------------------------

    for n = 1:numSensors

        % ------------------------------------------------------
        % Estimate delay kernel π_i(d)
        % ------------------------------------------------------
        % r_xy(d) counts how often sensor detections align
        % with oracle detections at delay d.

        r_xy = zeros(2*Delta+1,1);
        thetaFP_hat = 0;   % false positive rate

        for k = 1:T

            FP_flag = 1;

            if chi(y(k,n))

                for d = -Delta:Delta

                    kd = k + d;

                    if kd >= 1 && kd <= T && chi(x(kd))
                        FP_flag = 0;
                        r_xy(d+Delta+1) = r_xy(d+Delta+1) + 1;
                    end
                end
            else
                FP_flag = 0;
            end

            thetaFP_hat = thetaFP_hat + FP_flag;
        end

        TR = sum(r_xy);

        % Stabilize FP estimate
        thetaFP_hat = max(min(thetaFP_hat/T,0.99999),0.00001);

        % Normalize kernel
        if TR == 0
            % No reliable alignment → uniform kernel
            pi_hat = (1-thetaFP_hat) * ...
                     ones(size(r_xy)) / numel(r_xy);
        else
            pi_hat = (1-thetaFP_hat) * r_xy / TR;
        end

        %% ------------------------------------------------------
        % Log-likelihood computation
        % ------------------------------------------------------
        % Compute:
        %   p = Pr(Y(k)=0 | X, Δ)
        % and accumulate log-likelihood.

        q = ones(T,1);
        z = ones(T,1);

        for k = 1:T
            for d = -Delta:Delta

                kd = k + d;

                if kd >= 1 && kd <= T
                    z(k) = z(k) * (1 - chi(x(kd)));
                    q(k) = q(k) * (1 - pi_hat(d+Delta+1) * chi(x(kd)));
                end
            end
        end

        p = q - thetaFP_hat*z;

        logL(n) = logL(n) + sum( ...
            chi(y(:,n)) .* log(1-p) + ...
            (1-chi(y(:,n))) .* log(p));

        % Store estimated parameters
        model(n).pi      = pi_hat;
        model(n).thetaFP = thetaFP_hat;
        model(n).thetaV  = thetaV_hat;

        % ------------------------------------------------------
        % Information criterion (AIC)
        % ------------------------------------------------------

        criterion_score(1+Delta,n) = ...
            -2*logL(n) + 2*(2*Delta+3);
    end

    models{1+Delta} = model;
end

%% ==============================================================
% SELECT BEST Δ PER SENSOR
% ==============================================================

bestModel = cell(numSensors,1);
bestDelta = zeros(numSensors,1);

for n = 1:numSensors

    [~,bestDelta_] = min(criterion_score(:,n));

    bestModel{n} = models{bestDelta_}(n);
    bestDelta(n) = bestDelta_ - 1;
end


%% ==============================================================
% POSTERIOR ERROR ESTIMATION
% Compute Pr(sensor decision is wrong | data)
% ==============================================================

% err(k,n) = posterior probability that sensor n
% makes an incorrect decision at time k.

err = zeros(T, numSensors);

for n = 1:numSensors

    % Retrieve best model parameters for sensor n
    pi_hat       = bestModel{n}.pi;        % delay kernel
    thetaFP_hat  = bestModel{n}.thetaFP;   % false positive rate
    thetaV_hat   = bestModel{n}.thetaV;    % oracle activity rate
    Delta_n      = bestDelta(n);

    % Pre-compute q(k) = Pr(no true event influences time k)
    q = ones(T,1);

    for k = 1:T

        % Compute probability that no aligned oracle event
        % contributes to sensor detection at time k
        for d = -Delta_n:Delta_n

            kd = k + d;

            if kd >= 1 && kd <= T
                q(k) = q(k) * ...
                    (1 - pi_hat(d + Delta_n + 1) * chi(x(kd)));
            end
        end

        % ------------------------------------------------------
        % Posterior error computation via Bayes rule
        %
        % Two cases:
        %   1) Sensor reports detection
        %   2) Sensor reports no detection
        % ------------------------------------------------------

        if chi(y(k,n))
            % Case 1: sensor reports detection

            num_e = (1 - thetaV_hat) * thetaFP_hat;
            den_e = num_e + (1 - q(k)) * thetaV_hat;

        else
            % Case 2: sensor reports no detection

            num_e = q(k) * thetaV_hat;
            den_e = num_e + ...
                    (1 - thetaFP_hat) * (1 - thetaV_hat);
        end

        % Posterior probability of wrong detection
        err(k,n) = num_e / den_e;
    end
end


%% ==============================================================
% SENSOR ERROR SUMMARY
% ==============================================================

% Average posterior error per sensor (percentage)

avgErr = zeros(1, numSensors);

parfor n = 1:numSensors
    avgErr(n) = 100 * mean(err(:,n));
end

fprintf('\n');
disp('Average sensor errors [%]:');
disp(avgErr);

% Convert back to [0,1] scale for later use
avgErr = avgErr / 100;

% --------------------------------------------------------------
% Optional debugging:
% Uncomment to inspect signals and errors
% --------------------------------------------------------------
% [x y err]
% error('stop')


%% ==============================================================
% SENSOR MATCHING USING A TECHNIQUE FROM OPERATIONS RESEARCH
%
% Goal:
%   For each oracle vehicle (ground truth),
%   assign the best corresponding sensor detection.
%
% ==============================================================

% Oracle vehicle time instants
kx = find(x);                 
numVehs = numel(kx);          % total oracle vehicles

% Best solution containers (per sensor)
bestScore = -Inf * ones(numSensors,1);
bestW     = NaN(numSensors,1);
bestTimes = NaN(numVehs, numSensors);


numCat = 4;
EXPONENT = 1;
Penalty = 10^ceil(log10((1+numCat^EXPONENT)*numVehs));
weight = Penalty*ones(numVehs,T,numSensors); 
for n = 1:numSensors
    n

    W = bestDelta(n);
    timesArray = NaN(numVehs,1);

    % weight assignment
    matched = zeros(numVehs,1);
    for v = 1:numVehs
        for k = 1:T
            if abs(k-kx(v)) > W
                continue
            end
            if k == kx(v) && y(k,n) == x(k)
            %if y(k,n) == x(k)
                weight(v,k,n) = -1;
                matched(v) = 1; % perfect match
                %weight(v,k,n) = 0;
            elseif y(k,n) > 0
                % if you change the next line also change Penalty!
                weight(v,k,n) = err(k,n)+abs(y(k,n)-x(kx(v)))^EXPONENT;
                %weight(v,k,n) = abs(y(k,n)-x(kx(v)))^EXPONENT;
            end
        end
    end

    % Heuristics on weights for perfect matches:
    % increases n° of total mismatches but improves the quality of
    % associations established
    for v = 1:numVehs
        if matched(v)
            kv = kx(v);
            for w = [1:v-1 v+1:numVehs]
                weight(w,kv,n) = Penalty;
            end
        end
    end

    % shortest path
    Path = zeros(numVehs,1);
    Mat = Inf(T,numVehs);
    last = T-numVehs+1;
    Mat(1:last,1) = weight(1,1:last,n)';
    % for k = max(1,kx(1)-W):min(kx(1)+W,T-numVehs+1)
    %     Mat(k,v) = weight(1,k,n);
    % end
    for v = 1:numVehs-1
        [~,k] = min(Mat(:,v));
        Path(v) = k;
        last = T-numVehs+v;
        for k = v:last
            for kk = k+1:last+1
            %for kk = min(k+1,kx(v+1)-W):min(last+1,kx(v+1)+W)
                Mat(kk,v+1) = min(Mat(kk,v+1), Mat(k,v)+weight(v+1,kk,n));
            end
        end
    end
    [value,k] = min(Mat(:,numVehs));
    Path(numVehs) = k;

    % associations
    for v = 1:numVehs
        k = Path(v);
        if weight(v,k,n) < Penalty %&& y(k,n) > 0
            timesArray(v) = k;
        end
    end

    % ------------------------------------------------------
    % Objective function (in case multiple W are tested):
    % Maximize number of oracle vehicles detected
    % ------------------------------------------------------

    score = sum(~isnan(timesArray));

    % Keep best configuration
    if score > bestScore(n)
        bestScore(n)     = score;
        bestW(n)         = W;
        bestTimes(:,n)   = timesArray;
    end
    
end

%% ============================================================
% OUTPUT RESULTS
%% ============================================================

fprintf('\n=== SENSOR MATCHING OUTPUT ===\n');
fprintf('Videocamera\n')
fprintf('Number of real detections = %d\n', numVehs);
fprintf('Time instant: t = t_x0 + k,\n')
fprintf('              where t_x0 = %d,\n', tm)
fprintf('              and 1 <= k <= %d\n', T)

% Count occurrences per sensor
associations = sum(~isnan(bestTimes),1);
for n = 1:numSensors
    fprintf('\nSensor %2d\n', n);
    fprintf('Selected Delta = %d\n', bestDelta(n));
    fprintf('Optimal temporal window W = %d\n', bestW(n));
    fprintf('Number of associations = %d\n', associations(n));
    fprintf(['Time instant: t = t_y0' num2str(n) ' + k_hat,\n'])
    fprintf(['              where t_y0' num2str(n) ' = %d\n'], tm+mu(n))
end



% Detailed per-vehicle sensor assignments
fail_counter = 0;
Confusion_Matrix = zeros(numCat,numCat+1,numSensors);
fprintf('\n\nSensor matches per oracle vehicle:\n');
for v = 1:numVehs
    if sum(isnan(bestTimes(v,:))) == numSensors
        fail_counter = fail_counter + 1;
        fprintf('Vehicle %d (oracle k = %d): x = %d ------- \n',...
            v, kx(v), x(kx(v)));
    else
        fprintf('Vehicle %d (oracle k = %d): x = %d\n',...
            v, kx(v), x(kx(v)));
    end
    xx = x(kx(v));
    for n = 1:numSensors
        if ~isnan(bestTimes(v,n))
            fprintf('  Sensor %2d : k_hat = %d, y = %d\n',...
                n, bestTimes(v,n), y(bestTimes(v,n),n));
            yy = y(bestTimes(v,n),n);
            Confusion_Matrix(xx,yy,n) = Confusion_Matrix(xx,yy,n) + 1;
        else
            fprintf('  Sensor %2d : no detection\n', n);
            yy = numCat + 1;
            Confusion_Matrix(xx,yy,n) = Confusion_Matrix(xx,yy,n) + 1;
        end
    end
end

%% Confusion Matrix: compuation of Accuracy, Precision, Recall and Matthews correlation coefficient

Confusion_Matrix

Confusion_Matrix = Confusion_Matrix(:,1:numCat,:);

% Get dimensions of the hypermatrix
[num_rows, num_cols, num_slices] = size(Confusion_Matrix);

% Preallocate arrays to store global metrics for each slice (n = 1, 2, 3)
Accuracy_Global_All = zeros(num_slices, 1);
MCC_All = zeros(num_slices, 1);

% Loop through the third dimension (slices n = 1, 2, 3)
for n = 1:num_slices
    
    % Extract the current 2D confusion matrix slice
    Current_Matrix = Confusion_Matrix(:, :, n);
    
    % --- 1. GLOBAL ACCURACY CALCULATION ---
    TP_global = sum(diag(Current_Matrix)); 
    Total_Data = sum(Current_Matrix(:));  
    
    if Total_Data > 0
        Accuracy_Global_All(n) = TP_global / Total_Data;
    else
        Accuracy_Global_All(n) = 0;
    end

    % --- 2. PRECISION AND RECALL PER CLASS ---
    Precision_per_class = zeros(num_rows, 1);
    Recall_per_class = zeros(num_rows, 1);

    for i = 1:num_rows
        TP = Current_Matrix(i, i);
        Total_Row_Real = sum(Current_Matrix(i, :));
        Total_Col_Pred = sum(Current_Matrix(:, i));
        
        % Calculate Recall (handle division by zero)
        if Total_Row_Real > 0
            Recall_per_class(i) = TP / Total_Row_Real;
        else
            Recall_per_class(i) = 0;
        end
        
        % Calculate Precision (handle division by zero)
        if Total_Col_Pred > 0
            Precision_per_class(i) = TP / Total_Col_Pred;
        else
            Precision_per_class(i) = 0;
        end
    end

    % --- 3. MULTI-CLASS MATTHEWS CORRELATION COEFFICIENT (MCC) ---
    % Gorodkin generalization for rectangular or non-square matrices
    c = sum(diag(Current_Matrix));
    s = sum(Current_Matrix(:));
    pk = sum(Current_Matrix, 1);   % Column sums
    qk = sum(Current_Matrix, 2)';  % Row sums (transposed to row vector)

    % Expand qk to match pk dimensions if the matrix has an extra ghost column
    if num_cols > num_rows
        qk(num_cols) = 0; 
    end

    num = c * s - sum(pk .* qk);
    den = sqrt( (s^2 - sum(pk.^2)) * (s^2 - sum(qk.^2)) );

    if den == 0
        MCC_All(n) = 0;
    else
        MCC_All(n) = num / den;
    end

    % --- 4. DISPLAY RESULTS FOR THE CURRENT SLICE ---
    fprintf('==================================================\n');
    fprintf('       EVALUATION RESULTS FOR SLICE n = %d       \n', n);
    fprintf('==================================================\n');
    fprintf('Global Accuracy : %.2f%%\n', Accuracy_Global_All(n) * 100);
    fprintf('MCC             : %.4f\n\n', MCC_All(n));

    fprintf('Per-Class Breakdown:\n');
    for i = 1:num_rows
        fprintf('  Class %d -> Precision: %6.2f%% | Recall: %6.2f%%\n', ...
                i, Precision_per_class(i) * 100, Recall_per_class(i) * 100);
    end
    fprintf('==================================================\n\n');
end






%% ============================================================
% FIND REDUNDANT SENSOR SUBSETS
% Identify subsets of sensors that can be removed without
% decreasing the total number of vehicles detected
%% ============================================================

% Store subsets of removable sensors
removableSubsets = {};

% Total vehicles detected with all sensors
fullDetection = sum(~all(isnan(bestTimes),2));

% Loop over all possible subsets of sensors (power set)
for n = 1:numSensors  % subset size
    subsets = nchoosek(1:numSensors,n);
    for s = 1:size(subsets,1)
        removeSensors = subsets(s,:);  % candidate subset to remove
        
        % Temporary matrix with these sensors removed
        tempTimes = bestTimes;
        tempTimes(:,removeSensors) = NaN;
        
        % Count vehicles still detected
        carsDetected = sum(~all(isnan(tempTimes),2));
        
        % If unchanged, mark as removable
        if carsDetected == fullDetection
            removableSubsets{end+1} = removeSensors; 
        end
    end
end

%% ================================
% Display results
%% ================================

fprintf('\nRedundant sensor subsets (removable without total loss of association):\n');
if isempty(removableSubsets)
    fprintf('  None\n');
    fprintf('Total failures: %d\n', fail_counter);
else
    for s = 1:numel(removableSubsets)
        subset = removableSubsets{s};
        if isempty(subset)
            fprintf('  Empty subset (remove no sensors)\n');
        else
            fprintf('  Sensors: %s\n', mat2str(subset));
        end
    end
end



%% ============================================================
% CLASS MISMATCH ANALYSIS
%
% A mismatch is counted ONLY if:
%   - The sensor has an effective detection (bestTimes not NaN)
%   - The detected vehicle class differs from the oracle class
%
% For each sensor:
%   mismatchCount(n) ranges from 0 up to sensorCounts(n)
%% ============================================================

mismatchCount = zeros(1,numSensors);

for n = 1:numSensors
    
    count = 0;
    
    for v = 1:numVehs
        
        k_hat = bestTimes(v,n);
        
        % Consider only effective detections
        if ~isnan(k_hat)
            
            trueClass  = x(kx(v));       % oracle vehicle class
            sensorClass = y(k_hat,n);    % detected vehicle class
            
            % Count mismatch if classes differ
            if sensorClass ~= trueClass
                count = count + 1;
            end
        end
    end
    
    mismatchCount(n) = count;
end

% Display mismatch statistics
fprintf('\nVehicle class mismatches per sensor:\n');
for n = 1:numSensors
    fprintf('Sensor %2d: %d mismatches over %d associations\n', ...
        n, mismatchCount(n), associations(n));
end
























function [mu_hat, mu_all, J, Hhat, penalty] = ...
    estimate_lag_hh(Video, tm_x, tM_x, Sensor, tm_y, tM_y)
% ESTIMATE_LAG_HAMMING_HOEFFDING
%
% Estimates the minimum lag mu between two non-periodic binary time series
% using normalized Hamming distance with Hoeffding-based penalization.
%
% Penalization term:
%   lambda * sqrt(|L(mu)|)
%
% INPUT:
%   x, y   binary vectors (0/1) of equal length T
%
% OUTPUT:
%   mu_hat    estimated lag (minimum absolute value among minimizers)
%   mu_all    vector of all tested lags
%   J         penalized cost function values
%   Hhat      normalized Hamming distance
%   penalty   Hoeffding penalization term


    tm = max(tm_x,tm_y);
    tM = min(tM_x,tM_y);
    T = tM-tm+1;
    x = Video(tm:tM);
    y = Sensor(tm:tM);

    % confidence level (must be a value in (0,1))
    confidence = 0.95;

    % ------------------------
    % Define lag range
    % ------------------------
    mu_all = -(T-1):(T-1);
    L_mu_max = length(mu_all);

    Hhat = nan(L_mu_max, 1);
    penalty = nan(L_mu_max, 1);
    J = nan(L_mu_max, 1);

    % ------------------------
    % Loop over all lags
    % ------------------------
    for k = 1:L_mu_max
        mu = mu_all(k);

        if mu >= 0
            ix = 1:(T - mu);
            iy = (1 + mu):T;
        else
            ix = (1 - mu):T;
            iy = 1:(T + mu);
        end

        L_mu = length(ix);   % number of overlapping samples

        % Normalized Hamming distance
        Hhat(k) = mean(xor(x(ix), y(iy)));

        % lambda parameter (which is actually a function of |L(mu)|)
        lambda = sqrt(log(2/(1-confidence+2*confidence*exp(-2*L_mu)))/2);

        % Hoeffding-based uncertainty penalty
        penalty(k) = lambda / sqrt(L_mu);

        % Total cost
        J(k) = Hhat(k) + penalty(k);
    end

    % ------------------------
    % Select optimal lag
    % ------------------------
    Jmin = min(J);
    mu_candidates = mu_all(J == Jmin);

    % Choose the lag with minimum absolute value
    [~, idx] = min(abs(mu_candidates));
    mu_hat = mu_candidates(idx);
end


function value = chi(arg)
    value = ones(size(arg));
    for i = 1:length(arg)
        if arg(i) == 0
            value(i) = 0;
        end
    end
end


