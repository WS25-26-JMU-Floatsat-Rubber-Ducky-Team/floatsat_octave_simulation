addpath("utils");

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
  params.SIM_DURATION_S = 20.0;
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
  params.PID.Kp = [0.1, 0.1, 0.1];
  params.PID.Ki = [0.05, 0.05, 0.05];
  params.PID.Kd = [0.15, 0.15, 0.15];
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
  params.SETPOINT_DEG = [30, -5, 10]; % [roll, pitch, yaw] in degrees after step

  % Flywheel actuator parameters
  % per-wheel rotational inertia (kg·m^2). Measure or compute from geometry.
  params.FLYWHEEL.Jw = [0.005, 0.005, 0.005];
  % maximum wheel spin rate (rad/s)
  params.FLYWHEEL.wheel_angvel_max = [rpm2rads(900), rpm2rads(900), rpm2rads(900)]; % rad/s
  % motor torque limits
  params.FLYWHEEL.torque_max = [0.2, 0.2, 0.2]; % N·m (example values)

  % Geometry: wheel mount positions (r_i) and wheel spin axes (n_i) in body frame.
  % Each row is a vector [x,y,z]. These are separate physical quantities.
  params.WHEEL_POS = [ 0.0633, 0.0, 0.0;   % wheel 1 position (m)
                       0.0, 0.0633, 0.0;   % wheel 2 position
                       0.0, 0.0, -0.0633 ]; % wheel 3 position
  % Wheel spin axes (unit vectors). For tilted wheels set appropriate vectors (unitized).
  % params.WHEEL_AXIS = [ 0.8165, -0.4083, -0.4083;
  %                       -0.7071, 0.7071, -0.7071;
  %                       0.5773, 0.5773, -0.5773 ];

  params.WHEEL_AXIS = [ 1, 0, 0;
                        0, 1, 0;
                        0, 0, 1 ];

  params.I_body = [ 0.1, 0, 0;
                    0, 0.1, 0;
                    0, 0, 0.1 ];  % kg·m^2
  params.I_body_inv = inv(params.I_body);

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

  % Initial sensors from true state
  state.gyr_meas(1,:) = state.angvel_true(1,:);
  state.acc_meas(1,:) = true_accel_from_euler(state.angle_true(1,1), state.angle_true(1,2), state.angle_true(1,3), params.G);

  % Initial filter estimate from accel
  state.roll_f(1)  = atan2(state.acc_meas(1,2), state.acc_meas(1,3));
  state.pitch_f(1) = atan2(-state.acc_meas(1,1), sqrt(state.acc_meas(1,2)^2 + state.acc_meas(1,3)^2 + params.EPS));
  state.yaw_f(1)   = params.INITIAL_YAW;
  state.Qf(1,:) = eul2quat(state.roll_f(1), state.pitch_f(1), state.yaw_f(1));
  state.Qf(1,:) = normalize_quat(state.Qf(1,:));

  state.Q_true = state.Qf(1,:);  % true attitude quaternion

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
  EPS = params.EPS;

  for k = 2:N
    % Controller (PID) produces desired body torque (tau_desired_body)
    sp = state.setpoint(k,:);
    [rf, pf, yf, qf] = comp_step(state.roll_f(k-1), state.pitch_f(k-1), state.yaw_f(k-1), state.gyr_meas(k,:), state.acc_meas(k,:), params.comp);
    meas = [rf, pf, yf];
    % We'll treat PID output as a desired body torque (N·m). Convert units by tuning Ki/Kp accordingly.
    tau_desired_body = zeros(3,1);
    for axis = 1:3
      err = wrap_angle(sp(axis) - meas(axis));
      P = params.PID.Kp(axis) * err;
      state.pid.integrator(axis) = state.pid.integrator(axis) + err * DT * params.PID.Ki(axis);
      state.pid.integrator(axis) = clamp(state.pid.integrator(axis), state.PID_integrator_min, state.PID_integrator_max);
      I = state.pid.integrator(axis);
      derivative_raw = (err - state.pid.prev_error(axis)) / DT;
      tau = params.PID.tau_D;
      D_filtered = (tau * state.pid.prev_derivative(axis) + DT * derivative_raw) / (tau + DT);
      state.pid.prev_derivative(axis) = D_filtered;
      state.pid.prev_error(axis) = err;
      D = params.PID.Kd(axis) * D_filtered;
      % u is treated as desired body torque on that axis (N·m). Tune gains accordingly.
      u = P + I + D;
      tau_desired_body(axis) = u;
      state.cmd_angvel(k, axis) = 0; % keep for legacy plots; not used for dynamics now
    endfor

    % Torque allocation: map desired body torque to motor torques
    % Reaction torque from motor i is -tau_motor_i * n_i (n_i is wheel axis unit vector)
    % Solve: -N * tau_motor = tau_desired_body  => tau_motor = -pinv(N) * tau_desired_body
    % where N = [n1 n2 n3] (3x3), column i = axis_i
    Nmat = params.WHEEL_AXIS'; % 3x3 where column i is axis vector for wheel i
    % small regularization to avoid singularities
    reg = 1e-6;
    tau_motor = -pinv(Nmat' * Nmat + reg * eye(3)) * Nmat' * tau_desired_body;
    % clamp motor torques to motor limits
    for i = 1:3
      % tau_motor(i) = clamp(tau_motor(i), -params.FLYWHEEL.torque_max(i), params.FLYWHEEL.torque_max(i));
    endfor
    state.tau_motor_cmd(k,:) = tau_motor';

    % Integrate wheel dynamics: Jw * wdot = tau_motor - tau_fric (simple friction optional)
    Jw_vec = params.FLYWHEEL.Jw(:);
    wheel_w_prev = state.wheel_angvel(k-1, :)';
    wheel_w_dot = zeros(3,1);
    for i = 1:3
      % optional simple viscous friction coefficient (small)
      b = 1e-3;
      wheel_w_dot(i) = (tau_motor(i) - b * wheel_w_prev(i)) / Jw_vec(i);
      state.wheel_angvel(k,i) = wheel_w_prev(i) + wheel_w_dot(i) * DT;
      % enforce wheel speed limits
      state.wheel_angvel(k,i) = clamp(state.wheel_angvel(k,i), -params.FLYWHEEL.wheel_angvel_max(i), params.FLYWHEEL.wheel_angvel_max(i));
      state.wheel_angle(k,i) = state.wheel_angle(k-1,i) + state.wheel_angvel(k,i) * DT;
    endfor

    % Reaction torque on body from wheels (sum of -tau_motor_i * n_i)
    tau_from_wheels = zeros(3,1);
    for i = 1:3
      ni = params.WHEEL_AXIS(i,:)';
      tau_from_wheels = tau_from_wheels - tau_motor(i) * ni;
      % log per-wheel torques (desired/actual)
      state.torque_wheel_desired(k,i) = (-tau_desired_body' * ni); % projection for diagnostics
      state.torque_wheel_actual(k,i) = tau_motor(i);
    endfor

    % Integrate body rotational dynamics including gyroscopic term
    % I ω̇ + ω × (I ω) = τ_total  =>  ω̇ = I^{-1} ( τ_total - ω × (I ω) )
    % external torques are zero in this sim except wheel reaction torques
    omega_prev = state.angvel_true(k-1, :)';  % body rates in body frame
    I = params.I_body;
    Iomega = I * omega_prev;
    gyro_term = cross(omega_prev, Iomega);
    tau_total = tau_from_wheels; % plus any external torques if present
    omega_dot = params.I_body_inv * (tau_total - gyro_term);
    omega_new = omega_prev + omega_dot * DT;
    state.angvel_true(k, :) = omega_new';

    % Integrate attitude quaternion using body rates (body rates -> quaternion derivative)
    % q_dot = 0.5 * Omega(omega) * q
    q_prev = state.Qf(k-1,:)';
    % build Omega(omega) (4x4)
    wx = omega_new(1); wy = omega_new(2); wz = omega_new(3);
    Omega = 0.5 * [  0, -wx, -wy, -wz;
                    wx,   0,  wz, -wy;
                    wy, -wz,   0,  wx;
                    wz,  wy, -wx,   0 ];
    q_dot = Omega * q_prev;
    q_new = q_prev + q_dot * DT;
    q_new = q_new / norm(q_new + 1e-12);
    state.Qf(k,:) = q_new';
    % Update true Euler angles for logging/plots (convert quaternion -> euler)
    eul = quat2eul(q_new'); % assumes your eul2quat/quaternion convention matches quat2eul
    state.angle_true(k,:) = eul;
    state.roll_f(k) = eul(1); state.pitch_f(k) = eul(2); state.yaw_f(k) = eul(3);
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
  angle_true_deg = unwrap(deg(state.angle_true));
  angle_filt_deg = unwrap(deg([state.roll_f, state.pitch_f, state.yaw_f]));

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
  angvel_true_deg = deg(state.angvel_true);

  subplot(3,1,1);
  plot(t, cmd_before_deg(:,IDX_X), '--', 'LineWidth', 1.0); hold on;
  plot(t, angvel_true_deg(:,IDX_X), '-', 'LineWidth', 1.0);
  grid on; ylabel('Roll \omega (deg/s)');
  legend('cmd raw','true \omega','Location','NorthEast'); title('Roll angular velocity');

  subplot(3,1,2);
  plot(t, cmd_before_deg(:,IDX_Y), '--', 'LineWidth', 1.0); hold on;
  plot(t, angvel_true_deg(:,IDX_Y), '-', 'LineWidth', 1.0);
  grid on; ylabel('Pitch \omega (deg/s)');
  legend('cmd raw','true \omega','Location','NorthEast'); title('Pitch angular velocity');

  subplot(3,1,3);
  plot(t, cmd_before_deg(:,IDX_Z), '--', 'LineWidth', 1.0); hold on;
  plot(t, angvel_true_deg(:,IDX_Z), '-', 'LineWidth', 1.0);
  grid on; ylabel('Yaw \omega (deg/s)');
  xlabel('Time (s)');
  legend('cmd raw','true \omega','Location','NorthEast'); title('Yaw angular velocity');

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

function acc_body = true_accel_from_euler(roll, pitch, yaw, G)
  % Compute accelerometer reading in body frame due to gravity only.
  ax = -G * sin(pitch);
  ay =  G * sin(roll) * cos(pitch);
  az =  G * cos(roll) * cos(pitch);
  acc_body = [ax, ay, az];
endfunction

function [roll_f, pitch_f, yaw_f, qf] = comp_step(roll_prev, pitch_prev, yaw_prev, gyr, acc, params)
  % Quaternion-based complementary fusion:
  % - propagate quaternion using body rates (gyr)
  % - estimate roll/pitch from accel, build an accel-only quaternion (no yaw)
  % - slerp / blend between gyro-propagated quaternion and accel quaternion for roll/pitch correction
  DT = params.DT; ALPHA = params.ALPHA; EPS = params.EPS;

  % Reconstruct previous quaternion from euler inputs
  q_prev = eul2quat(roll_prev, pitch_prev, yaw_prev)';
  q_prev = q_prev / (norm(q_prev) + 1e-12);

  % Propagate quaternion using body rates (gyr)
  wx = gyr(params.IDX_X); wy = gyr(params.IDX_Y); wz = gyr(params.IDX_Z);
  Omega = 0.5 * [  0, -wx, -wy, -wz;
                  wx,   0,  wz, -wy;
                  wy, -wz,   0,  wx;
                  wz,  wy, -wx,   0 ];
  q_dot = Omega * q_prev;
  q_gyro = q_prev + q_dot * DT;
  q_gyro = q_gyro / (norm(q_gyro) + 1e-12);

  % Accel-derived roll/pitch (body frame)
  ax = acc(params.IDX_X); ay = acc(params.IDX_Y); az = acc(params.IDX_Z);
  roll_acc  = atan2(ay, az);
  pitch_acc = atan2(-ax, sqrt(ay*ay + az*az + EPS));
  % build accel quaternion (no yaw)
  q_acc = eul2quat(roll_acc, pitch_acc, 0)';
  q_acc = q_acc / (norm(q_acc) + 1e-12);

  % Complementary blend: prefer gyro for high-frequency, accel for low-frequency
  alpha = ALPHA;
  % Simple quaternion blend via normalized linear interpolation (approx slerp)
  qf = (alpha * q_gyro + (1 - alpha) * q_acc);
  qf = qf / (norm(qf) + 1e-12);

  % Output filtered Euler angles
  eul = quat2eul(qf'); % [roll pitch yaw]
  roll_f = wrap_angle(eul(1));
  pitch_f = wrap_angle(eul(2));
  yaw_f = wrap_angle(eul(3));
  qf = qf';
endfunction

function plot_3d_floatsat(state, params)
    figure('Name','3D FloatSat Visualization','NumberTitle','off');
    axis equal
    grid on
    xlabel('X'); ylabel('Y'); zlabel('Z');
    view(3)
    hold on

    % Fixed axes limits
    axis_limit = 0.2;
    axis([-axis_limit axis_limit -axis_limit axis_limit -axis_limit axis_limit]);
    axis manual;

    % Cube verts (body frame)
    s = 0.1;
    [Xc,Yc,Zc] = ndgrid([-1,1]*s/2);
    verts_cube = [Xc(:), Yc(:), Zc(:)];
    faces_cube = [1 3 7 5; 2 4 8 6; 1 2 6 5; 3 4 8 7; 1 2 4 3; 5 6 8 7];

    % Draw cube + top face handle(s)
    hCube = patch('Vertices',verts_cube,'Faces',faces_cube,'FaceColor','cyan','FaceAlpha',0.3);

    % Prepare flywheel meshes in local wheel coords (cylinder pointing along +Z)
    n_wheels = size(params.WHEEL_POS,1);
    wheel_mesh = cell(n_wheels,1);
    wheel_faces = cell(n_wheels,1);
    wheel_Ralign = cell(n_wheels,1);
    wheel_pos = cell(n_wheels,1);
    wheel_handle = zeros(1,n_wheels);

    % choose geometry for wheels
    wheel_radius = 0.03;
    wheel_length = 0.025;

    for i = 1:n_wheels
      [verts_cyl, faces_cyl] = make_cylinder_mesh(wheel_radius, wheel_length, 24);
      wheel_mesh{i} = verts_cyl; wheel_faces{i} = faces_cyl;
      ni = params.WHEEL_AXIS(i,:)';
      % precompute alignment rotation from +Z to ni (in body frame)
      % if ni is near [0;0;1], set Ralign=eye
      if norm(ni - [0;0;1]) < 1e-6
        Ralign = eye(3);
      else
        % rotation axis = cross(z, ni)
        rot_axis = cross([0;0;1], ni);
        s = norm(rot_axis);
        c = dot([0;0;1], ni);
        if s < 1e-9
          Ralign = eye(3);
        else
          Ralign = axis_angle_rot(rot_axis / s, atan2(s, c));
        end
      end
      wheel_Ralign{i} = Ralign;
      wheel_pos{i} = params.WHEEL_POS(i,:)'; % in body frame
      % create initial patch (will replace vertices every frame)
      wheel_handle(i) = patch('Vertices',verts_cyl,'Faces',faces_cyl,'FaceColor',[0.6,0.6,0.6],'FaceAlpha',1.0);
    endfor

    N = length(state.t);
    for k = 1:N
        q = state.Qf(k,:); % quaternion [w x y z]
        R_body_to_world = quat2rotm(q);   % 3x3 rotation matrix

        % Cube update
        verts_cube_world = (R_body_to_world * verts_cube')';
        set(hCube,'Vertices', verts_cube_world);

        % Wheels: transform and spin
        for i = 1:n_wheels
            verts_cyl = wheel_mesh{i}'; % 3 x Nv (in local cyl coords)
            % rotation about wheel axis by wheel_angle[k,i] (in body frame)
            angle = state.wheel_angle(k,i);
            ni = params.WHEEL_AXIS(i,:)';
            Rspin = axis_angle_rot(ni, angle); % rotate about ni in body frame
            % align cylinder then spin: produce verts in body frame
            verts_body = (Rspin * (wheel_Ralign{i} * verts_cyl));
            % translate to wheel mount pos (body frame)
            pos_body = wheel_pos{i};
            verts_body = verts_body + repmat(pos_body, 1, size(verts_body,2));
            % transform to world frame
            verts_world = (R_body_to_world * verts_body)';
            set(wheel_handle(i), 'Vertices', verts_world);
        endfor

        drawnow;
        pause(0.01);
    endfor
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

function [verts, faces] = make_cylinder_mesh(radius, length, nseg)
  % Cylinder along local Z from -length/2 .. +length/2
  if nargin < 3, nseg = 24; end
  [X,Y] = cylinder(radius, nseg);
  Z = repmat(linspace(-length/2, length/2, size(X,1))', 1, size(X,2));
  verts = [X(:), Y(:), Z(:)];
  % build faces (side quads + two caps)
  nv_per_ring = size(X,2);
  faces = [];
  % side quads
  for c = 1:(nv_per_ring-1)
    for r = 1:(size(X,1)-1)
      a = (r-1)*nv_per_ring + c;
      b = a + 1;
      d = a + nv_per_ring;
      cidx = b + nv_per_ring;
      faces = [faces; a b cidx d];
    endfor
  endfor
  % seam quads (wrap last->first)
  for r = 1:(size(X,1)-1)
    a = (r-1)*nv_per_ring + nv_per_ring;
    b = (r-1)*nv_per_ring + 1;
    d = a + nv_per_ring;
    cidx = b + nv_per_ring;
    faces = [faces; a b cidx d];
  endfor
  % caps (triangulate fan)
  top_center_idx = size(verts,1) + 1;
  bottom_center_idx = size(verts,1) + 2;
  verts = [verts; 0 0 length/2; 0 0 -length/2];
  for c = 1:(nv_per_ring-1)
    a = (size(X,1)-1)*nv_per_ring + c;
    b = a + 1;
    faces = [faces; a b top_center_idx top_center_idx];
  endfor
  % seam for top
  a = (size(X,1)-1)*nv_per_ring + nv_per_ring;
  b = (size(X,1)-1)*nv_per_ring + 1;
  faces = [faces; a b top_center_idx top_center_idx];
  % bottom cap (reverse order)
  for c = 1:(nv_per_ring-1)
    a = c;
    b = c + 1;
    faces = [faces; b a bottom_center_idx bottom_center_idx];
  endfor
  a = nv_per_ring;
  b = 1;
  faces = [faces; b a bottom_center_idx bottom_center_idx];
endfunction

function R = quat2rotm(q)
    % Convert quaternion [w x y z] to rotation matrix
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1-2*(y^2+z^2), 2*(x*y - z*w), 2*(x*z + y*w);
         2*(x*y + z*w), 1-2*(x^2+z^2), 2*(y*z - x*w);
         2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x^2+y^2)];
endfunction

% Run the main entry point automatically when the file is executed
main();
