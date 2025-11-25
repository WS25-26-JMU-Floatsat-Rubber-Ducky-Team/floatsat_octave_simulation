addpath("utils");

tilt_angles = 1:0.001:89;   % finer resolution for higher precision
sigmas = zeros(length(tilt_angles), 3);
conds  = zeros(length(tilt_angles), 1);

for k = 1:length(tilt_angles)
    [s, c, ~] = tilt_symmetric_wheel(tilt_angles(k));
    sigmas(k,:) = s';
    conds(k) = c;
endfor

% Find the tilt that gives the minimal condition number
[min_cond, idx_min] = min(conds);
optimal_tilt = tilt_angles(idx_min);
fprintf("Optimal tilt angle (deg) = %.5f, condition number = %.5f\n", optimal_tilt, min_cond);

% Plot singular values
fig1 = figure('Name','Singular Values vs Tilt','NumberTitle','off');
plot(tilt_angles, sigmas(:,1), 'r-', 'LineWidth',1.5); hold on;
plot(tilt_angles, sigmas(:,2), 'g-', 'LineWidth',1.5);
plot(tilt_angles, sigmas(:,3), 'b-', 'LineWidth',1.5);
grid on; xlabel('Tilt angle (deg)'); ylabel('Singular values');
legend('\sigma_1','\sigma_2','\sigma_3'); title('Torque Authority vs Tilt');
saveas(fig1, fullfile('img', 'singular_values_vs_tilt.png'));

% Plot condition number
fig2 = figure('Name','Condition Number vs Tilt','NumberTitle','off');
plot(tilt_angles, conds, 'k-', 'LineWidth',1.5); grid on;
xlabel('Tilt angle (deg)'); ylabel('Condition number');
title('Condition Number vs Tilt Angle');
ylim([0 3])

% Optional: mark the optimal tilt on the plot
hold on;
plot(optimal_tilt, min_cond, 'ro', 'MarkerSize',8, 'LineWidth',2);
text(optimal_tilt+1, min_cond, sprintf('%.3f°', optimal_tilt));
saveas(fig2, fullfile('img', 'condition_number_vs_tilt.png'));

pause();
