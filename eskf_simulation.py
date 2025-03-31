import numpy as np 
import matplotlib.pyplot as plt
import optuna

# ---------------------------
# Helper Functions (Quaternion Version)
# ---------------------------
def quaternion_mult(q, p):
    """
    Multiplies two quaternions.
    Both q and p are assumed to be [w, x, y, z].
    """
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_derivative(q, omega):
    """
    Computes the quaternion derivative given angular velocity omega.
    omega: angular velocity vector [ωx, ωy, ωz] in rad/s.
    The quaternion q is assumed to be [w, x, y, z].
    """
    omega_quat = np.concatenate([[0.0], omega])
    return 0.5 * quaternion_mult(q, omega_quat)

def normalize_quat(q):
    """
    Normalizes the quaternion to unit norm.
    """
    return q / np.linalg.norm(q)

def rotation_matrix(q):
    """
    Computes the rotation matrix from a unit quaternion.
    q: quaternion [w, x, y, z]
    Returns a 3x3 rotation matrix.
    """
    q = normalize_quat(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])
    return R

def cross(a, b):
    """Computes the cross product of 3D vectors a and b."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def skew(v):
    """Returns the skew-symmetric matrix of vector v."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],    0]])

def h_measurement(x):
    """
    Measurement function: returns the position part of the state.
    For the state: [q (4), velocity (3), position (3), sensor offset (3)],
    the position is x[7:10].
    """
    return x[7:10]

def sat(e, phi):
    """
    Saturation function that approximates the sign function with a boundary layer of thickness phi.
    """
    return np.clip(e, -phi, phi)

# ---------------------------
# IMU Simulation and Observer Class (Quaternion Version with ESKF)
# ---------------------------
class IMUSimulator:
    def __init__(self, dt=0.01, T_total=20.0, gps_rate=1.0, gps_delay=0.1,
                 gyro_noise_std=0.005, accel_noise_std=0.1, gps_noise_std=0.2,
                 r_true=None, seed=42):
        """
        Initializes simulation parameters.
        """
        np.random.seed(seed)
        self.dt = dt
        self.T_total = T_total
        self.steps = int(T_total / dt)
        self.time = np.linspace(0, T_total, self.steps)
        
        # Gravity in inertial frame
        self.g = np.array([0, 0, -9.81])
        
        # GPS parameters and true delay (converted to steps)
        self.gps_rate = gps_rate
        self.gps_update_steps = int(1.0 / (gps_rate * dt))
        self.gps_delay = gps_delay
        self.gps_delay_steps = int(gps_delay / dt)
        
        # True sensor offset (unknown to the observer)
        if r_true is None:
            self.r_true = np.array([0.05, -0.03, 0.10])
        else:
            self.r_true = np.array(r_true)
        
        # True initial state:
        # Orientation: identity quaternion [1,0,0,0]
        # Velocity and position zero, sensor offset as defined.
        self.q0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.v0 = np.array([0.0, 0.0, 0.0])
        self.p0 = np.array([0.0, 0.0, 0.0])
        self.x0_true = np.concatenate([self.q0, self.v0, self.p0, self.r_true])
        
        # Sensor noise parameters (external)
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.gps_noise_std = gps_noise_std

    def true_motion_profile(self, t):
        """
        Defines the true angular velocity, angular acceleration, and inertial acceleration.
        Uses a constant yaw rate and sinusoidal horizontal acceleration.
        """
        omega = np.array([0.0, 0.0, 0.1])
        dot_omega = np.array([0.0, 0.0, 0.0])
        a_inertial = np.array([0.5*np.cos(0.5*t), 0.5*np.sin(0.5*t), 0.0])
        return omega, dot_omega, a_inertial

    def true_a_body(self, x, t):
        """
        Computes the true accelerometer measurement in the body frame.
        """
        q = x[0:4]
        R = rotation_matrix(q)
        omega, dot_omega, _ = self.true_motion_profile(t)
        r = x[10:13]
        a_offset = R @ (cross(dot_omega, r) + cross(omega, cross(omega, r)))
        _, _, a_inertial = self.true_motion_profile(t)
        # To obtain the body-frame acceleration, invert the rotation.
        a_body = np.linalg.inv(R) @ (a_inertial + self.g + a_offset)
        return a_body

    def f_state(self, x, u, dt, g):
        """
        Nonlinear state transition function using quaternions.
        x: state vector (13,) = [q (4), velocity (3), position (3), sensor offset (3)]
        u: dictionary with keys: 'omega', 'a_body', 'dot_omega'
        dt: time step
        g: gravity vector
        """
        q = x[0:4]      # Quaternion [w, x, y, z]
        v = x[4:7]      # Velocity
        p = x[7:10]     # Position
        r = x[10:13]    # Sensor offset
        
        omega = u['omega']
        a_body = u['a_body']
        dot_omega = u['dot_omega']
        
        # Quaternion integration (first-order Euler)
        q_dot = quat_derivative(q, omega)
        q_new = q + dt * q_dot
        q_new = normalize_quat(q_new)
        
        # Compute rotation matrix from new quaternion
        R = rotation_matrix(q_new)
        
        # Sensor offset induced acceleration remains computed in inertial frame
        a_offset = R @ (cross(dot_omega, r) + cross(omega, cross(omega, r)))
        
        # Effective acceleration in inertial frame
        a_inertial = R @ a_body
        a_effective = a_inertial - g - a_offset
        
        # Update velocity and position
        v_new = v + a_effective * dt
        p_new = p + v * dt + 0.5 * a_effective * (dt**2)
        
        # Sensor offset remains constant
        r_new = r.copy()
        
        return np.concatenate([q_new, v_new, p_new, r_new])
    
    def simulate_true_trajectory(self):
        """
        Simulates the true state trajectory using the known dynamics.
        """
        n = 13  # state dimension
        x_true_hist = np.zeros((self.steps, n))
        x_true_hist[0] = self.x0_true
        for k in range(1, self.steps):
            t = self.time[k-1]
            omega, dot_omega, _ = self.true_motion_profile(t)
            u_true = {
                'omega': omega,
                'a_body': self.true_a_body(x_true_hist[k-1], t),
                'dot_omega': dot_omega
            }
            x_true_hist[k] = self.f_state(x_true_hist[k-1], u_true, self.dt, self.g)
        self.x_true_hist = x_true_hist
        return x_true_hist

    def simulate_measurements(self):
        """
        Simulates sensor measurements: high-rate gyro/accelerometer and low-rate, delayed GPS.
        """
        steps = self.steps
        time = self.time
        x_true_hist = self.x_true_hist
        
        z_gyro = np.zeros((steps, 3))
        z_accel = np.zeros((steps, 3))
        z_gps = np.full((steps, 3), np.nan)
        
        for k in range(steps):
            t = time[k]
            # Gyroscope measurement
            omega, _, _ = self.true_motion_profile(t)
            z_gyro[k] = omega + np.random.randn(3) * self.gyro_noise_std
            # Accelerometer measurement
            z_accel[k] = self.true_a_body(x_true_hist[k], t) + np.random.randn(3) * self.accel_noise_std
            # GPS measurement with delay (only after delay steps)
            if (k % self.gps_update_steps == 0) and (k >= self.gps_delay_steps):
                pos_true_delayed = x_true_hist[k - self.gps_delay_steps, 7:10]
                z_gps[k] = pos_true_delayed + np.random.randn(3) * self.gps_noise_std
        
        self.z_gyro = z_gyro
        self.z_accel = z_accel
        self.z_gps = z_gps
        return z_gyro, z_accel, z_gps

    def run_ESKF_with_delay_estimation(self, gains):
        """
        Runs an observer based on the Error-State Kalman Filter (ESKF) with delay compensation.
        In this design the nominal state is propagated nonlinearly using the IMU measurements,
        while a linearized error state (of dimension 12: [delta_theta, delta_v, delta_p, delta_r])
        is propagated using an approximate Jacobian. When a delayed GPS measurement is available,
        a measurement update is performed on the error state and the nominal state is corrected.
        
        The filter also updates an estimate of the GPS delay (tau_est) using a simple innovation-based rule.
        """
        n = 13  # nominal state dimension
        n_err = 12  # error state dimension (3 for delta_theta, 3 for delta_v, 3 for delta_p, 3 for delta_r)
        dt = self.dt
        steps = self.steps
        g = self.g
        epsilon = 1e-6

        # Initialize nominal state (same as before)
        x_hat = np.zeros(n)
        x_hat[0:4] = np.array([1.0, 0.0, 0.0, 0.0])
        x_hat[4:7] = self.v0
        x_hat[7:10] = self.p0
        x_hat[10:13] = np.zeros(3)
        
        # Initialize error covariance P (12x12)
        P = np.eye(n_err) * 1e-3

        # Initialize delay estimate tau_est (in seconds)
        tau_est = gains['tau0']
        tau_est_hist = np.zeros(steps)
        tau_est_hist[0] = tau_est

        # For delay compensation, store nominal state and covariance history
        x_hat_hist = np.zeros((steps, n))
        P_hist = np.zeros((steps, n_err, n_err))
        x_hat_hist[0] = x_hat
        P_hist[0] = P

        for k in range(1, steps):
            # Get sensor inputs at time k-1
            u_k = {
                'omega': self.z_gyro[k-1],
                'a_body': self.z_accel[k-1],
                'dot_omega': np.zeros(3)
            }
            # Prediction: integrate the nominal state using nonlinear dynamics
            x_hat_pred = self.f_state(x_hat, u_k, dt, g)

            # Compute a simple error state Jacobian F (12x12)
            F = np.zeros((n_err, n_err))
            F[0:3, 0:3] = -skew(self.z_gyro[k-1])
            R_nom = rotation_matrix(x_hat[0:4])
            F[3:6, 0:3] = - R_nom @ skew(self.z_accel[k-1])
            F[6:9, 3:6] = np.eye(3)
            # Sensor offset assumed constant, so F[9:12,9:12] = 0

            # Discrete-time state transition for the error state
            Phi = np.eye(n_err) + F * dt

            # Process noise covariance (discretized)
            Q = np.zeros((n_err, n_err))
            Q[0:3, 0:3] = np.eye(3) * (self.gyro_noise_std**2 * dt)
            Q[3:6, 3:6] = np.eye(3) * (self.accel_noise_std**2 * dt)
            Q_d = Q

            # Propagate error covariance
            P = Phi @ P @ Phi.T + Q_d

            # Update nominal state
            x_hat = x_hat_pred.copy()

            # Save history for delay compensation
            x_hat_hist[k] = x_hat
            P_hist[k] = P

            # If a delayed GPS measurement is available:
            if (not np.isnan(self.z_gps[k, 0])) and (k >= self.gps_delay_steps):
                # Determine the index corresponding to the current delay estimate.
                # Clip tau_est/dt between 0 and k to ensure a valid index.
                delay_steps_est = int(np.clip(tau_est / dt, 0, k))
                delayed_index = k - delay_steps_est

                # Retrieve the delayed nominal state and covariance
                x_delayed = x_hat_hist[delayed_index].copy()
                P_delayed = P_hist[delayed_index].copy()

                # Measurement: GPS measures position
                z = self.z_gps[k]
                p_pred = x_delayed[7:10]
                y = z - p_pred  # innovation

                # Measurement matrix H (maps error state to position error)
                H = np.zeros((3, n_err))
                H[:, 6:9] = np.eye(3)

                # Measurement noise covariance (GPS)
                R_meas = np.eye(3) * (self.gps_noise_std**2)

                # Kalman gain
                S = H @ P_delayed @ H.T + R_meas
                K = P_delayed @ H.T @ np.linalg.inv(S)

                # Error state update
                delta_x = K @ y

                # Correct the delayed nominal state:
                delta_theta = delta_x[0:3]
                delta_q = np.array([1.0, 0.5*delta_theta[0], 0.5*delta_theta[1], 0.5*delta_theta[2]])
                q_delayed = x_delayed[0:4]
                q_corr = quaternion_mult(q_delayed, delta_q)
                q_corr = normalize_quat(q_corr)
                v_corr = x_delayed[4:7] + delta_x[3:6]
                p_corr = x_delayed[7:10] + delta_x[6:9]
                r_corr = x_delayed[10:13] + delta_x[9:12]
                x_corr = np.concatenate([q_corr, v_corr, p_corr, r_corr])

                # Update covariance for the corrected delayed state
                P_corr = (np.eye(n_err) - K @ H) @ P_delayed

                # Propagate the corrected state forward from the delayed index to current time.
                x_update = x_corr.copy()
                P_update = P_corr.copy()
                for j in range(delayed_index+1, k+1):
                    u = {
                        'omega': self.z_gyro[j-1],
                        'a_body': self.z_accel[j-1],
                        'dot_omega': np.zeros(3)
                    }
                    x_update = self.f_state(x_update, u, dt, g)
                    q_temp = x_update[0:4]
                    R_temp = rotation_matrix(q_temp)
                    F_temp = np.zeros((n_err, n_err))
                    F_temp[0:3, 0:3] = -skew(self.z_gyro[j-1])
                    F_temp[3:6, 0:3] = - R_temp @ skew(self.z_accel[j-1])
                    F_temp[6:9, 3:6] = np.eye(3)
                    Phi_temp = np.eye(n_err) + F_temp * dt
                    Q_temp = np.zeros((n_err, n_err))
                    Q_temp[0:3, 0:3] = np.eye(3) * (self.gyro_noise_std**2 * dt)
                    Q_temp[3:6, 3:6] = np.eye(3) * (self.accel_noise_std**2 * dt)
                    P_update = Phi_temp @ P_update @ Phi_temp.T + Q_temp

                # Use the propagated corrected state as the current state
                x_hat = x_update.copy()
                P = P_update.copy()
                x_hat_hist[k] = x_hat
                P_hist[k] = P

                # Update delay estimate using the updated velocity
                vel_update = x_update[4:7]
                tau_est = tau_est + gains['gamma_tau'] * np.dot(vel_update, y) / (np.linalg.norm(vel_update)**2 + epsilon)
                # Clamp tau_est to be non-negative.
                tau_est = max(tau_est, 0)
                tau_est_hist[k] = tau_est
            else:
                tau_est_hist[k] = tau_est

        self.x_est_hist = x_hat_hist
        self.tau_est_hist = tau_est_hist
        self.tau_est_final = tau_est
        return x_hat_hist

    def run_simulation(self, gains, plot_results=False, use_ESKF=True):
        """
        Runs the full simulation: computes the true trajectory, simulates measurements,
        and applies the observer.
        If use_ESKF is True, the ESKF with delay compensation is used.
        Returns the RMSE in position.
        """
        self.simulate_true_trajectory()
        self.simulate_measurements()
        if use_ESKF:
            self.run_ESKF_with_delay_estimation(gains)
        else:
            # Fallback to previous observer (if desired)
            self.run_SMO_with_delay_estimation(gains)
        
        pos_true = self.x_true_hist[:, 7:10]
        pos_est = self.x_est_hist[:, 7:10]
        mse = np.mean(np.sum((pos_true - pos_est)**2, axis=1))
        rmse = np.sqrt(mse)
        
        if plot_results:
            self.plot_results(rmse)
            self.error_analysis()  # Perform error analysis
            print(f"Final estimated delay: {self.tau_est_final:.4f} s")
        return rmse

    def plot_results(self, rmse):
        """
        Generates plots for positions, velocities, accelerations, horizontal trajectory,
        and the evolution of the delay estimate.
        """
        dt = self.dt
        time = self.time
        pos_true = self.x_true_hist[:, 7:10]
        pos_est = self.x_est_hist[:, 7:10]
        vel_true = self.x_true_hist[:, 4:7]
        vel_est = self.x_est_hist[:, 4:7]
        gps_meas = self.z_gps
        
        # Compute acceleration (via finite differences)
        acc_true = np.diff(vel_true, axis=0) / dt
        acc_est = np.diff(vel_est, axis=0) / dt
        time_acc = time[1:]
        
        # ---- Positions ----
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(time, pos_true[:, 0], label='True x')
        axs[0].plot(time, pos_est[:, 0], '--', label='Estimated x')
        axs[0].plot(time, gps_meas[:, 0], 'o', markersize=3, label='GPS x', alpha=0.6)
        axs[0].set_ylabel('x (m)')
        axs[0].legend()
        axs[0].set_title("Position (x-component)")
        
        axs[1].plot(time, pos_true[:, 1], label='True y')
        axs[1].plot(time, pos_est[:, 1], '--', label='Estimated y')
        axs[1].plot(time, gps_meas[:, 1], 'o', markersize=3, label='GPS y', alpha=0.6)
        axs[1].set_ylabel('y (m)')
        axs[1].legend()
        axs[1].set_title("Position (y-component)")
        
        axs[2].plot(time, pos_true[:, 2], label='True z')
        axs[2].plot(time, pos_est[:, 2], '--', label='Estimated z')
        axs[2].plot(time, gps_meas[:, 2], 'o', markersize=3, label='GPS z', alpha=0.6)
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('z (m)')
        axs[2].legend()
        axs[2].set_title("Position (z-component)")
        fig.suptitle(f"True vs. Estimated vs. Measured Positions (RMSE = {rmse:.3f} m)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # ---- Velocities ----
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(time, vel_true[:, 0], label='True Vx')
        axs[0].plot(time, vel_est[:, 0], '--', label='Estimated Vx')
        axs[0].set_ylabel('Vx (m/s)')
        axs[0].legend()
        axs[0].set_title("Velocity (x-component)")
        
        axs[1].plot(time, vel_true[:, 1], label='True Vy')
        axs[1].plot(time, vel_est[:, 1], '--', label='Estimated Vy')
        axs[1].set_ylabel('Vy (m/s)')
        axs[1].legend()
        axs[1].set_title("Velocity (y-component)")
        
        axs[2].plot(time, vel_true[:, 2], label='True Vz')
        axs[2].plot(time, vel_est[:, 2], '--', label='Estimated Vz')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Vz (m/s)')
        axs[2].legend()
        axs[2].set_title("Velocity (z-component)")
        fig.suptitle("True vs. Estimated Velocities", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # ---- Accelerations ----
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(time_acc, acc_true[:, 0], label='True Ax')
        axs[0].plot(time_acc, acc_est[:, 0], '--', label='Estimated Ax')
        axs[0].set_ylabel('Ax (m/s²)')
        axs[0].legend()
        axs[0].set_title("Acceleration (x-component)")
        
        axs[1].plot(time_acc, acc_true[:, 1], label='True Ay')
        axs[1].plot(time_acc, acc_est[:, 1], '--', label='Estimated Ay')
        axs[1].set_ylabel('Ay (m/s²)')
        axs[1].legend()
        axs[1].set_title("Acceleration (y-component)")
        
        axs[2].plot(time_acc, acc_true[:, 2], label='True Az')
        axs[2].plot(time_acc, acc_est[:, 2], '--', label='Estimated Az')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Az (m/s²)')
        axs[2].legend()
        axs[2].set_title("Acceleration (z-component)")
        fig.suptitle("True vs. Estimated Accelerations (Finite Differences)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        # ---- Horizontal Trajectory ----
        plt.figure(figsize=(8, 6))
        plt.plot(pos_true[:, 0], pos_true[:, 1], label='True Trajectory')
        plt.plot(pos_est[:, 0], pos_est[:, 1], '--', label='Estimated Trajectory')
        valid = ~np.isnan(gps_meas[:, 0])
        plt.plot(pos_true[valid, 0], pos_true[valid, 1], 'o', markersize=3, label='GPS Measurements', alpha=0.6)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.legend()
        plt.title("Horizontal Trajectory (x-y)")
        plt.grid()
        plt.show()
        
        # ---- Delay Estimation ----
        plt.figure(figsize=(8, 4))
        plt.plot(self.time, self.tau_est_hist, label='Estimated Delay τ_est')
        plt.hlines(self.gps_delay, xmin=self.time[0], xmax=self.time[-1], colors='r', linestyles='--', label='True Delay')
        plt.xlabel("Time (s)")
        plt.ylabel("Delay (s)")
        plt.title("Evolution of Delay Estimate")
        plt.legend()
        plt.grid()
        plt.show()

    def error_analysis(self):
        """
        Performs error analysis on the simulation results.
        """
        time = self.time
        pos_true = self.x_true_hist[:, 7:10]
        pos_est = self.x_est_hist[:, 7:10]
        vel_true = self.x_true_hist[:, 4:7]
        vel_est = self.x_est_hist[:, 4:7]
        
        # Compute errors
        pos_error = pos_true - pos_est
        vel_error = vel_true - vel_est
        
        # Position error metrics
        pos_rmse = np.sqrt(np.mean(np.sum(pos_error**2, axis=1)))
        pos_bias = np.mean(pos_error, axis=0)
        pos_std = np.std(pos_error, axis=0)
        
        # Velocity error metrics
        vel_rmse = np.sqrt(np.mean(np.sum(vel_error**2, axis=1)))
        vel_bias = np.mean(vel_error, axis=0)
        vel_std = np.std(vel_error, axis=0)
        
        print("Error Analysis:")
        print(f"Position RMSE: {pos_rmse:.3f} m")
        print(f"Position Bias (mean error): {pos_bias}")
        print(f"Position Std Dev: {pos_std}")
        print("")
        print(f"Velocity RMSE: {vel_rmse:.3f} m/s")
        print(f"Velocity Bias (mean error): {vel_bias}")
        print(f"Velocity Std Dev: {vel_std}")
        
        # Plot norm of position error over time
        pos_error_norm = np.linalg.norm(pos_error, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(time, pos_error_norm, label='Position Error Norm')
        plt.xlabel("Time (s)")
        plt.ylabel("Error Norm (m)")
        plt.title("Position Error Norm over Time")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Plot norm of velocity error over time
        vel_error_norm = np.linalg.norm(vel_error, axis=1)
        plt.figure(figsize=(10, 4))
        plt.plot(time, vel_error_norm, label='Velocity Error Norm')
        plt.xlabel("Time (s)")
        plt.ylabel("Error Norm (m/s)")
        plt.title("Velocity Error Norm over Time")
        plt.legend()
        plt.grid()
        plt.show()
        
        # Histogram of position error norms
        plt.figure(figsize=(8, 4))
        plt.hist(pos_error_norm, bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel("Position Error Norm (m)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Position Error Norms")
        plt.grid()
        plt.show()
        
        # Delay estimation error analysis
        tau_error = self.tau_est_hist - self.gps_delay
        tau_rmse = np.sqrt(np.mean(tau_error**2))
        tau_bias = np.mean(tau_error)
        tau_std = np.std(tau_error)
        print("")
        print(f"Delay Estimation RMSE: {tau_rmse:.4f} s")
        print(f"Delay Estimation Bias: {tau_bias:.4f} s")
        print(f"Delay Estimation Std Dev: {tau_std:.4f} s")
        
        plt.figure(figsize=(10, 4))
        plt.plot(time, tau_error, label='Delay Estimation Error')
        plt.xlabel("Time (s)")
        plt.ylabel("Delay Error (s)")
        plt.title("Delay Estimation Error over Time")
        plt.legend()
        plt.grid()
        plt.show()

# ---------------------------
# Optuna Objective and Study (Tuning Observer Gains + Delay Estimation)
# ---------------------------
def objective(trial, params):
    gains = {
        'gamma_tau': trial.suggest_float('gamma_tau', 1e-3, 1.0, log=True),
        'tau0': trial.suggest_float('tau0', 0.01, 0.5)
    }
    simulator = IMUSimulator(
        dt=params['dt'],
        T_total=params['T_total'],
        gps_rate=params['gps_rate'],
        gps_delay=params['gps_delay'],
        gyro_noise_std=params['gyro_noise_std'],
        accel_noise_std=params['accel_noise_std'],
        gps_noise_std=params['gps_noise_std'],
        r_true=params['r_true']
    )
    rmse = simulator.run_simulation(gains, plot_results=False, use_ESKF=True)
    return rmse

# ---------------------------
# Main: Define Parameter Clusters and Run Optimization
# ---------------------------
if __name__ == '__main__':

    # 1. Consumer-Grade: Higher noise, lower update rates, lower accuracy.
    consumer_params = {
        'dt': 0.01,
        'T_total': 100.0,
        'gps_rate': 1.0,
        'gps_delay': 0.1,
        'gyro_noise_std': 0.05,
        'accel_noise_std': 0.2,
        'gps_noise_std': 2.0,
        'r_true': [0.1, 0.1, 0.1]
    }

    # 2. Industrial-Grade: Moderate noise and update rates.
    industrial_params = {
        'dt': 0.005,
        'T_total': 20.0,
        'gps_rate': 5.0,
        'gps_delay': 0.1,
        'gyro_noise_std': 0.005,
        'accel_noise_std': 0.05,
        'gps_noise_std': 0.5,
        'r_true': [0.05, -0.03, 0.10]
    }

    # 3. Tactical/Military-Grade: Very low sensor noise, high update rates.
    tactical_params = {
        'dt': 0.001,
        'T_total': 50.0,
        'gps_rate': 10.0,
        'gps_delay': 0.05,
        'gyro_noise_std': 0.001,
        'accel_noise_std': 0.01,
        'gps_noise_std': 0.1,
        'r_true': [0.01, 0.01, 0.01]
    }

    params = industrial_params  # Choose the desired parameter cluster

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, params), n_trials=50)
    
    print("Best trial:")
    print(f"  RMSE: {study.best_value:.3f}")
    print(f"  Gains & Delay Parameters: {study.best_params}")
    
    simulator = IMUSimulator(
        dt=params['dt'],
        T_total=params['T_total'],
        gps_rate=params['gps_rate'],
        gps_delay=params['gps_delay'],
        gyro_noise_std=params['gyro_noise_std'],
        accel_noise_std=params['accel_noise_std'],
        gps_noise_std=params['gps_noise_std'],
        r_true=params['r_true']
    )
    final_rmse = simulator.run_simulation(study.best_params, plot_results=True, use_ESKF=True)
    print(f"Final RMSE with optimized gains and delay estimation: {final_rmse:.3f}")
