# SensorFusion
This project tries various Kalman Filter methods to localize a mobile robot.
# 🛰️ Error-State Kalman Filter (ESKF) with Delay Compensation and Optuna Optimization

This repository implements an Error-State Kalman Filter (ESKF) observer designed for inertial navigation using IMU (gyroscope + accelerometer) and GPS measurements. It includes advanced features such as:

- Quaternion-based orientation tracking  
- Sensor offset modeling  
- Online estimation of GPS measurement delay  
- Full-state simulation environment  
- Optuna-based gain optimization  
- Visualization and error analysis tools

---

## 🚀 Features

- **Quaternion Kinematics:** Orientation is handled using quaternion algebra for robustness against singularities.  
- **Sensor Delay Estimation:** Estimates and compensates GPS measurement delay online.  
- **Optuna Integration:** Hyperparameter optimization for improved estimation performance.  
- **Customizable Simulation:** Configurable noise levels, GPS rates, and true sensor offsets.  
- **Extensive Plots and Error Metrics:** Analyze estimator accuracy through trajectory plots, RMSE, and bias metrics.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/eskf-delay-comp.git
cd eskf-delay-comp
pip install -r requirements.txt
```

**Dependencies:**

- numpy  
- matplotlib  
- optuna

---

## 📂 Project Structure

```
.
├── eskf_simulation.py     # Main script with ESKF implementation and simulator  
├── README.md              # This file  
└── results/               # (Optional) Output plots and metrics
```

---

## 🧠 How It Works

### State Representation

The nominal state vector includes:

```
x = [q (4), velocity (3), position (3), sensor_offset (3)]
```

The error-state vector used for Kalman updates:

```
δx = [δθ (3), δv (3), δp (3), δr (3)]
```

### Measurements

- **IMU:** High-rate gyroscope and accelerometer (with additive noise).  
- **GPS:** Low-rate, delayed position updates (with known delay).

### Delay Compensation

The ESKF tracks the history of state estimates. When a delayed GPS measurement is received, the filter:

1. Retrieves the delayed state estimate.  
2. Applies the Kalman correction.  
3. Propagates the corrected state forward to the current time.  
4. Updates the estimated GPS delay using innovation-based feedback.

---

## 🧪 Running the Simulation

```bash
python eskf_simulation.py
```

This will:

- Simulate the IMU and GPS data  
- Run the ESKF with delay estimation  
- Optimize filter gains using Optuna  
- Display detailed plots and error metrics

---

## ⚙️ Tuning Parameters

Inside `__main__`, you can customize the simulation:

```python
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
```

And Optuna will optimize these gains:

```python
gains = {
    'gamma_tau': [1e-3, 1.0],  # Delay estimation gain
    'tau0': [0.01, 0.5]        # Initial guess for GPS delay
}
```

---

## 📊 Output

- Position, velocity, acceleration plots  
- Horizontal trajectory overlay  
- Delay estimate vs. true delay  
- RMSE, bias, standard deviation  
- Histograms and error norms

---

## 📈 Optimization with Optuna

Optuna automatically tunes the observer’s delay estimation parameters (`gamma_tau`, `tau0`) to minimize RMSE in position.

```python
study.optimize(lambda trial: objective(trial, params), n_trials=50)
```

After tuning, it reruns the simulation with the best configuration.

---

## 🛠️ Future Improvements

- Support for magnetometer or barometer fusion  
- Adaptive noise covariance tuning  
- 3D trajectory visualization  
- ROS integration

---

## 📜 License

MIT License © 2025 Your Name

---

## 🤝 Contributions

Issues and pull requests are welcome! If you find a bug or want to add features, feel free to fork and submit.
