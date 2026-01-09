addpath("utils");
addpath("visualization");

function main()
  % High-level entry: setup, simulate, plot
  params = setup_params();
  state  = setup_state(params);
  [state, params] = run_simulation(state, params);
  plot_results(state, params);
  plot_3d_floatsat(state, params);
  pause();
endfunction

% -------------------------------------------------------------------------
% Params and initialization
% -------------------------------------------------------------------------
function params = setup_params()
  % Simulation params
  params.SAMPLE_RATE_HZ = 100;
  params.DT = 1.0 / params.SAMPLE_RATE_HZ;
  params.SIM_DURATION_S = 30.0;
  params.N = round(params.SIM_DURATION_S * params.SAMPLE_RATE_HZ);

  % Physical constants
  params.G = 9.81;
  params.EPS = 1e-8;

  % Axis indices
  params.IDX_X = 1; params.IDX_Y = 2; params.IDX_Z = 3;

  % Initial yaw
  params.INITIAL_YAW = 0.0;

  % PID tuning (per-axis)
  params.PID.Kp = [1.5, 1.5, 1.5];
  params.PID.Ki = [3, 3, 3];
  params.PID.Kd = [5, 5, 5];
  params.PID.tau_D = 0.02;
  params.PID.integrator_min_frac = -0.5;  % fraction of actuator max (scaled later)
  params.PID.integrator_max_frac =  0.5;

  % Actuator (placeholder: currently maps commanded angvel -> true angvel)
  params.ACTUATOR_MAX_ANGVEL_DEG = 60;
  params.ACTUATOR_MAX_ANGVEL = deg2rad(params.ACTUATOR_MAX_ANGVEL_DEG);
  params.ACTUATOR_MIN_ANGVEL = -params.ACTUATOR_MAX_ANGVEL;
  params.ACTUATOR_TIMECONST = 0.05;

  % Setpoint: change at 1 second
  params.SETPOINT_STEP_TIME = 1.0;
  params.SETPOINT_DEG = [-10, 10, 40]; % [roll, pitch, yaw] in degrees after step

  % Flywheel actuator parameters
  % per-wheel rotational inertia (kg·m^2). Measure or compute from geometry.
  params.FLYWHEEL.Jw = [0.01, 0.01, 0.01];
  % maximum wheel spin rate (rad/s)
  params.FLYWHEEL.wheel_angvel_max = [rpm2rads(900), rpm2rads(900), rpm2rads(900)]; % rad/s
  % motor torque limits
  params.FLYWHEEL.torque_max = [0.2, 0.2, 0.2]; % N·m (example values)

  azimuth_deg = [0, 120, 240];   % wheel azimuths in XY plane
  azimuth_rad = deg2rad(azimuth_deg);

  r_mount = 0.0633;   % distance from body center in XY plane
  z_offset = -0.03022; % slightly below body center along Z

  wheel_pos = zeros(3,3); % 3 wheels x 3 components
  wheel_axis = zeros(3,3);

  for i = 1:3
      az = azimuth_rad(i);
      wheel_pos(i,1) = r_mount * cos(az);  % X
      wheel_pos(i,2) = r_mount * sin(az);  % Y
      wheel_pos(i,3) = z_offset;            % Z
  end

  for i = 1:3
      vec = -wheel_pos(i,:);  % vector from wheel to center
      wheel_axis(i,:) = vec / norm(vec);  % normalize
  end

  params.WHEEL_POS  = wheel_pos;
  params.WHEEL_AXIS = wheel_axis;

  % params.WHEEL_POS = [ 0.063, 0, 0;
  %                      0, 0.063, 0;
  %                      0, 0, -0.063 ];
  % params.WHEEL_AXIS = [ 1, 0, 0;
  %                       0, 1, 0;
  %                       0, 0, 1 ];

  params.I_body = [ 1, 0, 0;
                    0, 1, 0;
                    0, 0, 1 ];  % kg·m^2
  params.I_body_inv = inv(params.I_body);

  % Make params easier to pass to comp_step
  params.comp.DT = params.DT;
  params.comp.EPS = params.EPS;
  params.comp.IDX_X = params.IDX_X;
  params.comp.IDX_Y = params.IDX_Y;
  params.comp.IDX_Z = params.IDX_Z;

  params.SENSOR.MAG_DIR = [1; 0; 0]; % direction of magnetic north in inertial coords (unit)
  params.SENSOR.MAG_NOISE_STD = 0.05;      % white noise std per sample (rad/s)
  params.SENSOR.MAG_BIAS_INIT_STD = 0.01;   % initial bias std (rad/s)
  params.SENSOR.MAG_BIAS_RW_STD = 1e-4;     % bias random-walk std (rad/s / sqrt(s))

  params.SENSOR.GYRO_NOISE_STD = 0.05;      % white noise std per sample (rad/s)
  params.SENSOR.GYRO_BIAS_INIT_STD = 0.01;   % initial bias std (rad/s)
  params.SENSOR.GYRO_BIAS_RW_STD = 1e-4;     % bias random-walk std (rad/s / sqrt(s))

  params.SENSOR.ACC_NOISE_STD = 0.05;        % white noise std per sample (m/s^2)
  params.SENSOR.ACC_BIAS_INIT_STD = 0.1;     % initial accel bias std (m/s^2)
  params.SENSOR.ACC_BIAS_RW_STD = 1e-3;      % accel bias random-walk std (m/s / sqrt(s))

  % Optional: quantization (e.g., ADC or sensor LSB). Set to 0 to disable.
  params.SENSOR.MAG_QUANT = 0.0;   % rad/s
  params.SENSOR.GYRO_QUANT = 0.0;   % rad/s
  params.SENSOR.ACC_QUANT  = 0.0;   % m/s^2

  % Complementary filter tuning
  params.TAU = 0.5;
  params.TAU_YAW = 5.0;

  params.FILTER.WEIGHT_ACC = params.DT / params.TAU;     % roll/pitch correction
  params.FILTER.WEIGHT_MAG = params.DT / params.TAU_YAW; % yaw correction

endfunction

function state = setup_state(params)
  N = params.N;
  IDX_X = params.IDX_X; IDX_Y = params.IDX_Y; IDX_Z = params.IDX_Z;
  DT = params.DT;

  % Preallocate logs & state
  state.t = (0:(N-1))' * DT;
  state.setpoint = zeros(N,3);
  for k = 1:N
    if ( (k-1)*DT >= params.SETPOINT_STEP_TIME )
      state.setpoint(k,:) = deg2rad(params.SETPOINT_DEG);  % [roll,pitch,yaw] in rad
    end
  endfor

  state.angle_true = zeros(N,3);
  state.angvel_true = zeros(N,3);
  state.angvel_act_state = zeros(1,3);

  state.roll_f = zeros(N,1); state.pitch_f = zeros(N,1); state.yaw_f = zeros(N,1);
  state.Qf = zeros(N,4);

  state.gyr_meas = zeros(N,3);
  state.acc_meas = zeros(N,3);
  state.mag_meas = zeros(N,3);

  % Measured Euler angles (noisy) stored in state for easy plotting
  state.angle_meas = zeros(N,3);   % [roll, pitch, yaw] measured (rad)

  state.cmd_angvel = zeros(N,3);
  state.cmd_after_act = zeros(N,3);

  % PID state
  state.pid.integrator = [0,0,0];
  state.pid.prev_error = [0,0,0];
  state.pid.prev_derivative = [0,0,0];

  % Wheel torque applied at each step
  state.torque_wheel_actual = zeros(N,3);   % torque actually applied by wheel
  state.torque_wheel_desired = zeros(N,3);  % PID-desired torque on body

  % Wheel speeds over time (N×3) and motor torque command history
  state.wheel_angvel = zeros(N,3);         % rad/s, wheel spin rates logged
  state.tau_motor_cmd = zeros(N,3);        % commanded motor torque (N·m)
  state.wheel_angle  = zeros(N,3);         % rad, integrated wheel rotation (for visualization)
  % initial wheel speed (0)
  state.wheel_angvel(1,:) = [0,0,0];
  state.wheel_angle(1,:) = [0,0,0];

  % Initial true state: level + initial yaw
  state.angle_true(1,:) = [0, 0, params.INITIAL_YAW];
  state.angvel_true(1,:) = [0,0,0];

  if isfield(params, 'RNG_SEED') && ~isempty(params.RNG_SEED)
    rand("seed", params.RNG_SEED);
  end

  % True sensor logs (for analysis)
  state.gyr_true = zeros(N,3);   % true angular rate
  state.acc_true = zeros(N,3);   % true accel

  % Bias state (kept in state so we can log it)
  state.mag_bias  = zeros(N,3);  % per-step bias history: starts from random initial bias
  state.gyro_bias = zeros(N,3);
  state.acc_bias  = zeros(N,3);

  % Initialize first-step biases (draw from initial bias std)
  state.mag_bias(1,:) = params.SENSOR.MAG_BIAS_INIT_STD * randn(1,3);
  state.gyro_bias(1,:) = params.SENSOR.GYRO_BIAS_INIT_STD * randn(1,3);
  state.acc_bias(1,:)  = params.SENSOR.ACC_BIAS_INIT_STD  * randn(1,3);

  % Initial filter estimate from accel
  state.roll_f(1)  = atan2(state.acc_meas(1,2), state.acc_meas(1,3));
  state.pitch_f(1) = atan2(-state.acc_meas(1,1), sqrt(state.acc_meas(1,2)^2 + state.acc_meas(1,3)^2 + params.EPS));
  state.yaw_f(1)   = params.INITIAL_YAW;
  state.Qf(1,:) = eul2quat(state.roll_f(1), state.pitch_f(1), state.yaw_f(1));
  state.Qf(1,:) = normalize_quat(state.Qf(1,:));

  % true attitude quaternion history (store per-step)
  state.Q_true = zeros(N,4);
  state.Q_true(1,:) = state.Qf(1,:);

  % Initial sensors from true state
  state.gyr_meas(1,:) = state.angvel_true(1,:);
  state.acc_meas(1,:) = true_accel_from_quat(state.Q_true, params.G);

  % Scale integrator limits based on actuator
  state.PID_integrator_min = params.PID.integrator_min_frac * 1;
  state.PID_integrator_max = params.PID.integrator_max_frac * 1;
endfunction

% -------------------------------------------------------------------------
% Simulation loop
% -------------------------------------------------------------------------
function [state, params] = run_simulation(state, params)
  N = params.N;
  DT = params.DT;

  for k = 2:N
    % Form noisy sensor measurements for time k based on previous true state q_prev and angvel at k-1
    q_prev_f = state.Qf(k-1,:)';      % previous filtered quaternion (4x1)
    q_prev_t = state.Q_true(k-1,:)';  % previous true quaternion (4x1)
    % true sensors from previous true/est state:
    acc_true_prev = true_accel_from_quat(q_prev_t', params.G); % 1x3
    gyr_true_prev = state.angvel_true(k-1,:);                 % 1x3

    % evolve biases (random walk) from k-1 -> k
    gyro_bias_rw_per_step = params.SENSOR.GYRO_BIAS_RW_STD * sqrt(DT);
    acc_bias_rw_per_step  = params.SENSOR.ACC_BIAS_RW_STD  * sqrt(DT);
    state.gyro_bias(k,:) = state.gyro_bias(k-1,:) + gyro_bias_rw_per_step * randn(1,3);
    state.acc_bias(k,:)  = state.acc_bias(k-1,:)  + acc_bias_rw_per_step  * randn(1,3);

    % white noise
    gyr_noise = params.SENSOR.GYRO_NOISE_STD * randn(1,3);
    acc_noise = params.SENSOR.ACC_NOISE_STD  * randn(1,3);
    mag_noise = params.SENSOR.MAG_NOISE_STD * randn(1,3);

    % true mag in body frame
    R_ib = quat2rotm(q_prev_t');    % body->inertial; transpose = inertial->body
    m_body_true = R_ib' * params.SENSOR.MAG_DIR; % 3x1
    mag_bias_rw_per_step = params.SENSOR.MAG_BIAS_RW_STD * sqrt(DT); % add bias + noise
    state.mag_bias(k,:) = state.mag_bias(k-1,:) + mag_bias_rw_per_step * randn(1,3);

    mag_noise = params.SENSOR.MAG_NOISE_STD * randn(3,1);

    % measured signals (bias + noise)
    gyr_meas = gyr_true_prev + state.gyro_bias(k,:) + gyr_noise;
    acc_meas = acc_true_prev + state.acc_bias(k,:) + acc_noise;
    mag_meas = m_body_true + state.mag_bias(k,:)' + mag_noise;

    % optional quantization
    if params.SENSOR.MAG_QUANT > 0
      mag_meas = round(mag_meas / params.SENSOR.MAG_QUANT) * params.SENSOR.MAG_QUANT;
    end
    if params.SENSOR.GYRO_QUANT > 0
      gyr_meas = round(gyr_meas / params.SENSOR.GYRO_QUANT) * params.SENSOR.GYRO_QUANT;
    end
    if params.SENSOR.ACC_QUANT > 0
      acc_meas = round(acc_meas / params.SENSOR.ACC_QUANT) * params.SENSOR.ACC_QUANT;
    end

    % store noisy measurements for this timestep (so the filter sees them)
    state.gyr_meas(k,:) = gyr_meas;
    state.acc_meas(k,:) = acc_meas;
    state.mag_meas(k,:) = mag_meas';

    % compute measured roll/pitch from accel_meas (same as filter accel branch)
    ax = acc_meas(params.IDX_X);
    ay = acc_meas(params.IDX_Y);
    az = acc_meas(params.IDX_Z);
    roll_meas  = atan2(ay, az);
    pitch_meas = atan2(-ax, sqrt(ay*ay + az*az + params.EPS));

    % tilt-compensated magnetometer using the *noisy* mag_meas (column vector)
    mx = mag_meas(1); my = mag_meas(2); mz = mag_meas(3);
    sin_r = sin(roll_meas); cos_r = cos(roll_meas);
    sin_p = sin(pitch_meas); cos_p = cos(pitch_meas);

    mx2 = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p;
    my2 = my * cos_r - mz * sin_r;

    % IMPORTANT: use same sign convention as quat_from_accmag (atan2(-my2, mx2))
    yaw_meas = atan2(-my2, mx2);

    % store measured Euler angles (rad)
    state.angle_meas(k, :) = [roll_meas, pitch_meas, yaw_meas];

    % (optional) store true sensors too for logging
    state.gyr_true(k,:) = gyr_true_prev;
    state.acc_true(k,:) = acc_true_prev;
    qf = comp_step_quat(q_prev_f, state.gyr_meas(k,:), state.acc_meas(k,:), state.mag_meas(k,:), params);
    state.Qf(k,:) = qf';

    eul_filt = quat2eul(qf');  % [roll pitch yaw]
    state.roll_f(k)  = eul_filt(1);
    state.pitch_f(k) = eul_filt(2);
    state.yaw_f(k)   = eul_filt(3);

    % --- Step 2: Compute quaternion setpoint from setpoint Euler angles ---
    q_setpoint = eul2quat(state.setpoint(k,1), state.setpoint(k,2), state.setpoint(k,3))';

    % --- Step 3: PID in quaternion error space ---
    [tau_desired_body, state.pid] = pid_quat(qf, q_setpoint, state.pid, params, params.DT);

    % --- Step 4: Torque allocation to wheels ---
    Nmat = params.WHEEL_AXIS'; % 3x3 where column i = wheel axis vector
    reg = 1e-6; % regularization to avoid singularities
    tau_motor = -pinv(Nmat' * Nmat + reg*eye(3)) * Nmat' * tau_desired_body;

    % Clamp motor torques to limits
    for i = 1:3
      tau_motor(i) = clamp(tau_motor(i), -params.FLYWHEEL.torque_max(i), params.FLYWHEEL.torque_max(i));
    end
    state.tau_motor_cmd(k,:) = tau_motor';

    % --- Step 5: Integrate wheel dynamics ---
    Jw_vec = params.FLYWHEEL.Jw(:);
    wheel_w_prev = state.wheel_angvel(k-1,:)';
    wheel_w_dot = zeros(3,1);
    for i = 1:3
      b = 1e-3; % small viscous friction
      wheel_w_dot(i) = (tau_motor(i) - b*wheel_w_prev(i)) / Jw_vec(i);
      state.wheel_angvel(k,i) = wheel_w_prev(i) + wheel_w_dot(i)*DT;
      % enforce wheel speed limits
      state.wheel_angvel(k,i) = clamp(state.wheel_angvel(k,i), -params.FLYWHEEL.wheel_angvel_max(i), params.FLYWHEEL.wheel_angvel_max(i));
      % integrate wheel angle for visualization
      state.wheel_angle(k,i) = state.wheel_angle(k-1,i) + state.wheel_angvel(k,i)*DT;
    endfor

    % --- Step 6: Compute reaction torque from wheels ---
    tau_from_wheels = zeros(3,1);
    for i = 1:3
      ni = params.WHEEL_AXIS(i,:)';
      tau_from_wheels = tau_from_wheels - tau_motor(i) * ni;
      state.torque_wheel_desired(k,i) = (-tau_desired_body' * ni); % diagnostic
      state.torque_wheel_actual(k,i) = tau_motor(i);
    endfor

    % --- Step 7: Integrate body rotational dynamics ---
    omega_prev = state.angvel_true(k-1,:)';
    I = params.I_body;
    gyro_term = cross(omega_prev, I*omega_prev);
    omega_dot = params.I_body_inv * (tau_from_wheels - gyro_term);
    omega_new = omega_prev + omega_dot*DT;
    state.angvel_true(k,:) = omega_new';

    % --- Step 8: Update true quaternion using body rates ---
    wx = omega_new(1); wy = omega_new(2); wz = omega_new(3);
    Omega = 0.5 * [  0, -wx, -wy, -wz;
                     wx,  0,  wz, -wy;
                     wy, -wz,  0,  wx;
                     wz,  wy, -wx,  0 ];
    q_dot_t = Omega * q_prev_t;
    q_new_t = q_prev_t + q_dot_t*DT;
    q_new_t = q_new_t / norm(q_new_t + 1e-12);

    % store true quaternion separately (do NOT overwrite filtered quaternion)
    state.Q_true(k,:) = q_new_t';

    % Update Euler angles for plotting only
    eul = quat2eul(q_new_t'); % convert for plots
    state.angle_true(k,:) = eul;
  endfor
  % Print mean absolute angle error (deg)
  true_angles  = state.angle_true;                     % Nx3 (rad)
  filt_angles  = [state.roll_f, state.pitch_f, state.yaw_f];  % Nx3 (rad)

  angle_err_rad = abs(filt_angles - true_angles);
  angle_err_deg = (180/pi) * angle_err_rad;

  mean_err_deg = mean(angle_err_deg, 1);   % 1x3

  fprintf('\nMean absolute angle error:\n');
  fprintf('  Roll  error: %.4f deg\n', mean_err_deg(1));
  fprintf('  Pitch error: %.4f deg\n', mean_err_deg(2));
  fprintf('  Yaw   error: %.4f deg\n', mean_err_deg(3));
  fprintf('\n');
endfunction

% -------------------------------------------------------------------------
% Plotting helper
% -------------------------------------------------------------------------
function plot_results(state, params)
  IDX_X = params.IDX_X; IDX_Y = params.IDX_Y; IDX_Z = params.IDX_Z;
  t = state.t;
  deg = @(x) (180/pi) * x;

  setpoint_deg = deg(state.setpoint);
  angle_true_deg = deg(state.angle_true);
  angle_meas_deg = deg(state.angle_meas);
  angle_filt_deg = deg([state.roll_f, state.pitch_f, state.yaw_f]);

  % Angles: 3 subplots (Setpoint / True / Measured / Filtered)
  figure('Name','Angles: Setpoint vs True vs Measured vs Filtered','NumberTitle','off');

  subplot(3,1,1);
  plot(t, setpoint_deg(:,IDX_X), '--', 'LineWidth', 1.2); hold on;
  plot(t, angle_true_deg(:,IDX_X), '-', 'LineWidth', 1.0);
  plot(t, angle_meas_deg(:,IDX_X), '-.', 'LineWidth', 1.0);
  plot(t, angle_filt_deg(:,IDX_X), ':', 'LineWidth', 1.2);
  grid on; ylabel('Roll (deg)');
  legend('setpoint','true','measured','filtered','Location','NorthEast');
  title('Roll');

  subplot(3,1,2);
  plot(t, setpoint_deg(:,IDX_Y), '--', 'LineWidth', 1.2); hold on;
  plot(t, angle_true_deg(:,IDX_Y), '-', 'LineWidth', 1.0);
  plot(t, angle_meas_deg(:,IDX_Y), '-.', 'LineWidth', 1.0);
  plot(t, angle_filt_deg(:,IDX_Y), ':', 'LineWidth', 1.2);
  grid on; ylabel('Pitch (deg)');
  legend('setpoint','true','measured','filtered','Location','NorthEast');
  title('Pitch');

  subplot(3,1,3);
  plot(t, setpoint_deg(:,IDX_Z), '--', 'LineWidth', 1.2); hold on;
  plot(t, angle_true_deg(:,IDX_Z), '-', 'LineWidth', 1.0);
  plot(t, angle_meas_deg(:,IDX_Z), '-.', 'LineWidth', 1.0);
  plot(t, angle_filt_deg(:,IDX_Z), ':', 'LineWidth', 1.2);
  grid on; ylabel('Yaw (deg)'); xlabel('Time (s)');
  legend('setpoint','true','measured','filtered','Location','NorthEast');
  title('Yaw');
  % Angular velocity commands
  figure('Name','Angular velocity commands','NumberTitle','off');
  cmd_before_deg = deg(state.cmd_angvel);
  angvel_true_deg = deg(state.angvel_true);

  subplot(3,1,1);
  plot(t, cmd_before_deg(:,IDX_X), '--', 'LineWidth', 1.0); hold on;
  plot(t, angvel_true_deg(:,IDX_X), '-', 'LineWidth', 1.0);
  grid on; ylabel('Roll \omega (deg/s)');
  legend('cmd raw','true \omega','Location','NorthEast'); title('Flywheel Angular Velocity: Roll');

  subplot(3,1,2);
  plot(t, cmd_before_deg(:,IDX_Y), '--', 'LineWidth', 1.0); hold on;
  plot(t, angvel_true_deg(:,IDX_Y), '-', 'LineWidth', 1.0);
  grid on; ylabel('Pitch \omega (deg/s)');
  legend('cmd raw','true \omega','Location','NorthEast'); title('Flywheel Angular Velocity: Pitch');

  subplot(3,1,3);
  plot(t, cmd_before_deg(:,IDX_Z), '--', 'LineWidth', 1.0); hold on;
  plot(t, angvel_true_deg(:,IDX_Z), '-', 'LineWidth', 1.0);
  grid on; ylabel('Yaw \omega (deg/s)');
  xlabel('Time (s)');
  legend('cmd raw','true \omega','Location','NorthEast'); title('Flywheel Angular Velocity: Yaw');

  figure('Name','Flywheel Torque','NumberTitle','off');
  torque_desired_deg = state.torque_wheel_desired;
  torque_actual_deg = state.torque_wheel_actual;

  subplot(3,1,1);
  plot(state.t, torque_desired_deg(:,params.IDX_X), '--', 'LineWidth', 1.2); hold on;
  plot(state.t, torque_actual_deg(:,params.IDX_X), '-', 'LineWidth', 1.2);
  grid on; ylabel('Roll Torque (N/m)'); title('Flywheel Torque: Roll');
  legend('desired','actual','Location','NorthEast');

  subplot(3,1,2);
  plot(state.t, torque_desired_deg(:,params.IDX_Y), '--', 'LineWidth', 1.2); hold on;
  plot(state.t, torque_actual_deg(:,params.IDX_Y), '-', 'LineWidth', 1.2);
  grid on; ylabel('Pitch Torque (N/m)'); title('Flywheel Torque: Pitch');
  legend('desired','actual','Location','NorthEast');

  subplot(3,1,3);
  plot(state.t, torque_desired_deg(:,params.IDX_Z), '--', 'LineWidth', 1.2); hold on;
  plot(state.t, torque_actual_deg(:,params.IDX_Z), '-', 'LineWidth', 1.2);
  grid on; ylabel('Yaw Torque (N/m)'); xlabel('Time (s)'); title('Flywheel Torque: Yaw');
  legend('desired','actual','Location','NorthEast');

  drawnow();
endfunction

% -------------------------------------------------------------------------
% Reusable utility functions
% -------------------------------------------------------------------------

function acc_body = true_accel_from_quat(q, G)
  % Compute accelerometer reading in body frame due to gravity only,
  % given quaternion q = [w x y z] (body -> inertial).
  %
  % acc_body is 1x3 row vector [ax ay az].
  %
  % G is gravity magnitude (positive scalar, e.g. 9.81). If your inertial
  % gravity vector points negative Z, call with G = -9.81 or adjust below.
  if size(q,2) ~= 4 && size(q,1) == 4
    q = q'; % make row
  end

  % Gravity in inertial frame (pointing +Z). If your convention uses -Z,
  % change this to [0;0;-G].
  g_inertial = [0; 0; G];

  % Rotation matrix body->inertial
  R_ib = quat2rotm(q);
  % Rotation from inertial -> body is R_ib'
  acc_body_vec = R_ib' * g_inertial;

  acc_body = acc_body_vec(:)'; % return as 1x3 row vector
endfunction

function qf = comp_step_quat(q_prev, gyr, acc, mag, params)
  DT = params.DT;
  EPS = params.EPS;

  % Gyro propagation
  wx = gyr(1); wy = gyr(2); wz = gyr(3);
  Omega = 0.5 * [  0, -wx, -wy, -wz;
                   wx,   0,  wz, -wy;
                   wy, -wz,   0,  wx;
                   wz,  wy, -wx,   0 ];
  q_dot = Omega * q_prev;
  q_gyro = q_prev + q_dot * DT;
  q_gyro = q_gyro / (norm(q_gyro) + 1e-12);

  % Extract gyro Euler angles
  eul_gyro = quat2eul(q_gyro');  % [roll pitch yaw]

  roll_g  = eul_gyro(1);
  pitch_g = eul_gyro(2);
  yaw_g   = eul_gyro(3);

  % Roll/pitch from accelerometer
  ax = acc(1); ay = acc(2); az = acc(3);
  roll_acc  = atan2(ay, az);
  pitch_acc = atan2(-ax, sqrt(ay^2 + az^2 + EPS));

  % Yaw from magnetometer (tilt compensated)
  sin_r = sin(roll_g);  cos_r = cos(roll_g);
  sin_p = sin(pitch_g); cos_p = cos(pitch_g);

  mx = mag(1); my = mag(2); mz = mag(3);
  mx2 = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p;
  my2 = my * cos_r - mz * sin_r;

  yaw_mag = atan2(-my2, mx2);

  % Weighted complementary correction
  roll_f  = (1 - params.FILTER.WEIGHT_ACC) * roll_g  + params.FILTER.WEIGHT_ACC * roll_acc;
  pitch_f = (1 - params.FILTER.WEIGHT_ACC) * pitch_g + params.FILTER.WEIGHT_ACC * pitch_acc;

  % wrap-safe yaw blending
  yaw_err = atan2(sin(yaw_mag - yaw_g), cos(yaw_mag - yaw_g));
  yaw_f   = yaw_g + params.FILTER.WEIGHT_MAG * yaw_err;

  % --- 6. Back to quaternion ---
  qf = eul2quat(roll_f, pitch_f, yaw_f)';
  qf = qf / (norm(qf) + 1e-12);
endfunction

function q_accmag = quat_from_accmag(acc, mag, EPS)
  % Compute quaternion from accelerometer + magnetometer
  % Returns 4x1 column vector
  ax = acc(1); ay = acc(2); az = acc(3);
  mx = mag(1); my = mag(2); mz = mag(3);

  % Roll/pitch from accel
  roll  = atan2(ay, az);
  pitch = atan2(-ax, sqrt(ay^2 + az^2 + EPS));

  % Tilt-compensated magnetometer
  sin_r = sin(roll); cos_r = cos(roll);
  sin_p = sin(pitch); cos_p = cos(pitch);

  mx2 = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p;
  my2 = my * cos_r - mz * sin_r;

  yaw = atan2(-my2, mx2);

  % Euler -> quaternion
  q_accmag = eul2quat(roll, pitch, yaw)';
  q_accmag = q_accmag / (norm(q_accmag) + 1e-12);
endfunction

function q_interp = slerp(q1, q2, t)
  % Spherical linear interpolation between quaternions q1->q2
  % q1, q2 are 4x1 column vectors
  dotp = dot(q1, q2);
  if dotp < 0
      q2 = -q2; dotp = -dotp;  % take shortest path
  end
  dotp = min(max(dotp, -1), 1);  % clamp for acos
  theta = acos(dotp);
  if theta < 1e-6
      q_interp = q1;  % nearly identical
      return;
  end
  q_interp = (sin((1-t)*theta)/sin(theta))*q1 + (sin(t*theta)/sin(theta))*q2;
endfunction

function R = axis_angle_rot(axis, theta)
  % Rodrigues rotation matrix: rotate around (3x1) axis (unit) by theta radians.
  if norm(axis) < 1e-12
    R = eye(3);
    return;
  end
  u = axis / norm(axis);
  ux = [   0, -u(3),  u(2);
         u(3),    0, -u(1);
        -u(2), u(1),    0 ];
  R = eye(3) * cos(theta) + (1 - cos(theta)) * (u * u') + ux * sin(theta);
endfunction

function [tau_desired_body, pid_state] = pid_quat(q_current, q_setpoint, pid_state, params, DT)
  % Compute body torque using quaternion error
  % q_current, q_setpoint: 4x1 quaternions
  % pid_state: struct with .integrator, .prev_error (3x1)
  % Returns 3x1 torque vector

  % --- Quaternion error (rotation from current -> setpoint) ---
  q_err = quatmultiply(quatconjugate(q_current'), q_setpoint'); % 1x4
  if q_err(1) < 0
      q_err = -q_err;  % shortest rotation
  end
  % rotation vector approximation: small angle
  rot_err = q_err(2:4) * 2;  % 3x1 vector

  tau_desired_body = zeros(3,1);
  for axis = 1:3
      err = rot_err(axis);
      % PID
      pid_state.integrator(axis) = pid_state.integrator(axis) + err * DT * params.PID.Ki(axis);
      pid_state.integrator(axis) = clamp(pid_state.integrator(axis), ...
                                         params.PID.integrator_min_frac, ...
                                         params.PID.integrator_max_frac);
      derivative_raw = (err - pid_state.prev_error(axis)) / DT;
      tau = params.PID.tau_D;
      D_filtered = (tau * pid_state.prev_derivative(axis) + DT * derivative_raw) / (tau + DT);
      pid_state.prev_derivative(axis) = D_filtered;
      pid_state.prev_error(axis) = err;
      tau_desired_body(axis) = params.PID.Kp(axis)*err + pid_state.integrator(axis) + params.PID.Kd(axis)*D_filtered;
  end
endfunction

% Run the main entry point automatically when the file is executed
main();
