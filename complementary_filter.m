addpath("utils");

function main()
  % High-level entry: setup, simulate, plot
  params = setup_params();
  state  = setup_state(params);
  [state, params] = run_simulation(state, params);
  plot_results(state, params);
  pause();
endfunction

% -------------------------------------------------------------------------
% Params and initialization
% -------------------------------------------------------------------------
function params = setup_params()
  % Simulation params
  params.SAMPLE_RATE_HZ = 100;
  params.DT = 1.0 / params.SAMPLE_RATE_HZ;
  params.SIM_DURATION_S = 4.0;
  params.N = round(params.SIM_DURATION_S * params.SAMPLE_RATE_HZ);

  % Physical constants
  params.G = 9.81;
  params.EPS = 1e-8;

  % Axis indices
  params.IDX_X = 1; params.IDX_Y = 2; params.IDX_Z = 3;

  % Complementary filter tuning (time constant -> alpha)
  params.TAU = 0.5;
  params.ALPHA = params.TAU / (params.TAU + params.DT);

  % Initial yaw
  params.INITIAL_YAW = 0.0;

  % PID tuning (per-axis)
  params.PID.Kp = [4.0, 4.0, 4.0];
  params.PID.Ki = [1.0, 1.0, 1.0];
  params.PID.Kd = [0.1, 0.1, 0.1];
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
  params.SETPOINT_DEG = [10, -5, 20]; % [roll, pitch, yaw] in degrees after step

  % Make params easier to pass to comp_step
  params.comp.DT = params.DT;
  params.comp.ALPHA = params.ALPHA;
  params.comp.EPS = params.EPS;
  params.comp.IDX_X = params.IDX_X;
  params.comp.IDX_Y = params.IDX_Y;
  params.comp.IDX_Z = params.IDX_Z;
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
      state.setpoint(k,:) = deg2rad(params.SETPOINT_DEG);
    endif
  endfor

  state.angle_true = zeros(N,3);
  state.angvel_true = zeros(N,3);
  state.angvel_act_state = zeros(1,3);

  state.roll_f = zeros(N,1); state.pitch_f = zeros(N,1); state.yaw_f = zeros(N,1);
  state.Qf = zeros(N,4);

  state.gyr_meas = zeros(N,3);
  state.acc_meas = zeros(N,3);

  state.cmd_angvel = zeros(N,3);
  state.cmd_after_act = zeros(N,3);

  % PID state
  state.pid.integrator = [0,0,0];
  state.pid.prev_error = [0,0,0];
  state.pid.prev_derivative = [0,0,0];

  % Initial true state: level + initial yaw
  state.angle_true(1,:) = [0, 0, params.INITIAL_YAW];
  state.angvel_true(1,:) = [0,0,0];

  % Initial sensors from true state
  state.gyr_meas(1,:) = state.angvel_true(1,:);
  state.acc_meas(1,:) = true_accel_from_euler(state.angle_true(1,1), state.angle_true(1,2), state.angle_true(1,3), params.G);

  % Initial filter estimate from accel
  state.roll_f(1)  = atan2(state.acc_meas(1,2), state.acc_meas(1,3));
  state.pitch_f(1) = atan2(-state.acc_meas(1,1), sqrt(state.acc_meas(1,2)^2 + state.acc_meas(1,3)^2 + params.EPS));
  state.yaw_f(1)   = params.INITIAL_YAW;
  state.Qf(1,:) = eul2quat(state.roll_f(1), state.pitch_f(1), state.yaw_f(1));
  state.Qf(1,:) = normalize_quat(state.Qf(1,:));

  % Scale integrator limits based on actuator
  state.PID_integrator_min = params.PID.integrator_min_frac * params.ACTUATOR_MAX_ANGVEL;
  state.PID_integrator_max = params.PID.integrator_max_frac * params.ACTUATOR_MAX_ANGVEL;
endfunction

% -------------------------------------------------------------------------
% Simulation loop
% -------------------------------------------------------------------------
function [state, params] = run_simulation(state, params)
  N = params.N;
  DT = params.DT;
  EPS = params.EPS;

  for k = 2:N
    % 1) Sensors sample the true plant
    state.gyr_meas(k,:) = state.angvel_true(k-1,:);
    state.acc_meas(k,:) = true_accel_from_euler(state.angle_true(k-1,1), state.angle_true(k-1,2), state.angle_true(k-1,3), params.G);

    % 2) Filter: fused estimate from sensors
    [rf, pf, yf, qf] = comp_step(state.roll_f(k-1), state.pitch_f(k-1), state.yaw_f(k-1), state.gyr_meas(k,:), state.acc_meas(k,:), params.comp);
    state.roll_f(k) = rf; state.pitch_f(k) = pf; state.yaw_f(k) = yf; state.Qf(k,:) = qf;

    % 3) Controller (PID) uses the FILTERED measurements
    sp = state.setpoint(k,:);
    meas = [rf, pf, yf];
    for axis = 1:3
      err = wrap_angle(sp(axis) - meas(axis));

      % P
      P = params.PID.Kp(axis) * err;

      % I (with clamping)
      state.pid.integrator(axis) = state.pid.integrator(axis) + err * DT * params.PID.Ki(axis);
      % clamp integrator
      state.pid.integrator(axis) = clamp(state.pid.integrator(axis), state.PID_integrator_min, state.PID_integrator_max);
      I = state.pid.integrator(axis);

      % D (filtered derivative)
      derivative_raw = (err - state.pid.prev_error(axis)) / DT;
      tau = params.PID.tau_D;
      D_filtered = (tau * state.pid.prev_derivative(axis) + DT * derivative_raw) / (tau + DT);
      state.pid.prev_derivative(axis) = D_filtered;
      state.pid.prev_error(axis) = err;
      D = params.PID.Kd(axis) * D_filtered;

      % PID -> commanded angular velocity (rad/s)
      u = P + I + D;
      state.cmd_angvel(k, axis) = u;
    endfor

    % 4) Actuator (currently simple angvel actuator; swap this for flywheel later)
    for axis = 1:3
      u_sat = clamp(state.cmd_angvel(k,axis), params.ACTUATOR_MIN_ANGVEL, params.ACTUATOR_MAX_ANGVEL);
      if params.ACTUATOR_TIMECONST <= EPS
        state.angvel_act_state(axis) = u_sat;
      else
        alpha_act = DT / (params.ACTUATOR_TIMECONST + DT);
        state.angvel_act_state(axis) = state.angvel_act_state(axis) + alpha_act * (u_sat - state.angvel_act_state(axis));
      endif
      state.cmd_after_act(k, axis) = state.angvel_act_state(axis);
      state.angvel_true(k, axis) = state.angvel_act_state(axis);
    endfor

    % 5) Plant integration (angles)
    state.angle_true(k, :) = state.angle_true(k-1, :) + state.angvel_true(k, :) * DT;
    state.angle_true(k,1) = wrap_angle(state.angle_true(k,1));
    state.angle_true(k,2) = wrap_angle(state.angle_true(k,2));
    state.angle_true(k,3) = wrap_angle(state.angle_true(k,3));
  endfor
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
  angle_filt_deg = deg([state.roll_f, state.pitch_f, state.yaw_f]);

  % Angles: 3 subplots
  figure('Name','Angles: Setpoint vs True vs Filtered','NumberTitle','off');
  subplot(3,1,1);
  plot(t, setpoint_deg(:,IDX_X), '--', 'LineWidth', 1.2); hold on;
  plot(t, angle_true_deg(:,IDX_X), '-', 'LineWidth', 1.0);
  plot(t, angle_filt_deg(:,IDX_X), ':', 'LineWidth', 1.2);
  grid on; ylabel('Roll (deg)'); legend('setpoint','true','filtered','Location','NorthEast'); title('Roll');

  subplot(3,1,2);
  plot(t, setpoint_deg(:,IDX_Y), '--', 'LineWidth', 1.2); hold on;
  plot(t, angle_true_deg(:,IDX_Y), '-', 'LineWidth', 1.0);
  plot(t, angle_filt_deg(:,IDX_Y), ':', 'LineWidth', 1.2);
  grid on; ylabel('Pitch (deg)'); legend('setpoint','true','filtered','Location','NorthEast'); title('Pitch');

  subplot(3,1,3);
  plot(t, setpoint_deg(:,IDX_Z), '--', 'LineWidth', 1.2); hold on;
  plot(t, angle_true_deg(:,IDX_Z), '-', 'LineWidth', 1.0);
  plot(t, angle_filt_deg(:,IDX_Z), ':', 'LineWidth', 1.2);
  grid on; ylabel('Yaw (deg)'); xlabel('Time (s)'); legend('setpoint','true','filtered','Location','NorthEast'); title('Yaw');

  % Angular velocity commands
  figure('Name','Angular velocity commands','NumberTitle','off');
  cmd_before_deg = deg(state.cmd_angvel);
  cmd_after_deg = deg(state.cmd_after_act);
  angvel_true_deg = deg(state.angvel_true);

  subplot(3,1,1);
  plot(t, cmd_before_deg(:,IDX_X), '--', 'LineWidth', 1.0); hold on;
  plot(t, cmd_after_deg(:,IDX_X), '-', 'LineWidth', 1.0);
  plot(t, angvel_true_deg(:,IDX_X), ':', 'LineWidth', 1.0);
  grid on; ylabel('Roll \omega (deg/s)');
  legend('cmd raw','cmd after act','true \omega','Location','NorthEast'); title('Roll angular velocity');

  subplot(3,1,2);
  plot(t, cmd_before_deg(:,IDX_Y), '--', 'LineWidth', 1.0); hold on;
  plot(t, cmd_after_deg(:,IDX_Y), '-', 'LineWidth', 1.0);
  plot(t, angvel_true_deg(:,IDX_Y), ':', 'LineWidth', 1.0);
  grid on; ylabel('Pitch \omega (deg/s)');
  legend('cmd raw','cmd after act','true \omega','Location','NorthEast'); title('Pitch angular velocity');

  subplot(3,1,3);
  plot(t, cmd_before_deg(:,IDX_Z), '--', 'LineWidth', 1.0); hold on;
  plot(t, cmd_after_deg(:,IDX_Z), '-', 'LineWidth', 1.0);
  plot(t, angvel_true_deg(:,IDX_Z), ':', 'LineWidth', 1.0);
  grid on; ylabel('Yaw \omega (deg/s)');
  xlabel('Time (s)');
  legend('cmd raw','cmd after act','true \omega','Location','NorthEast'); title('Yaw angular velocity');

  drawnow();
endfunction

% -------------------------------------------------------------------------
% Reusable utility functions
% -------------------------------------------------------------------------

function acc_body = true_accel_from_euler(roll, pitch, yaw, G)
  % Compute accelerometer reading in body frame due to gravity only.
  ax = -G * sin(pitch);
  ay =  G * sin(roll) * cos(pitch);
  az =  G * cos(roll) * cos(pitch);
  acc_body = [ax, ay, az];
endfunction

function [roll_f, pitch_f, yaw_f, qf] = comp_step(roll_prev, pitch_prev, yaw_prev, gyr, acc, params)
  DT = params.DT; ALPHA = params.ALPHA; EPS = params.EPS;
  ix = params.IDX_X; iy = params.IDX_Y; iz = params.IDX_Z;

  roll_gyro  = roll_prev  + gyr(ix) * DT;
  pitch_gyro = pitch_prev + gyr(iy) * DT;
  yaw_gyro   = yaw_prev   + gyr(iz) * DT;

  ax = acc(ix); ay = acc(iy); az = acc(iz);
  denom = sqrt(ay*ay + az*az + EPS);
  roll_acc  = atan2(ay, az);
  pitch_acc = atan2(-ax, denom);

  roll_f  = ALPHA * roll_gyro  + (1 - ALPHA) * roll_acc;
  pitch_f = ALPHA * pitch_gyro + (1 - ALPHA) * pitch_acc;
  yaw_f   = yaw_gyro;

  roll_f  = wrap_angle(roll_f);
  pitch_f = wrap_angle(pitch_f);
  yaw_f   = wrap_angle(yaw_f);

  qf = eul2quat(roll_f, pitch_f, yaw_f);
  qf = normalize_quat(qf);
endfunction

% Run the main entry point automatically when the file is executed
main();
