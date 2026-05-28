# ============================================================
# FOUR-STROKE OTTO-CYCLE SLIDER-CRANK WITH FLYWHEEL
# Full simulation, settled-cycle comparison and optional animation
# ============================================================
# Changes made for clearer flywheel results:
#   - The flywheel comparison is no flywheel vs I_f = 1.0 kg m^2.
#   - c_rot is increased from 0.05 to 0.10 N m s/rad to represent a
#     stronger external load and allow periodic operating behaviour to
#     be reached in a practical simulation time.
#   - Results are calculated over complete late Otto cycles, not over
#     the whole startup transient.
#   - The time of the largest recorded speed is still reported separately.
#
# The selected flywheel is a conceptual solid disk for this large-scale
# model: m = 12.5 kg, R = 0.40 m, so I_f = 1.0 kg m^2.
# ============================================================

import csv
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# ============================================================
# 1. SETTINGS
# ============================================================
OUTPUT_DIR = Path('flywheel_results')
OUTPUT_DIR.mkdir(exist_ok=True)

SIMULATION_END_TIME = 30.0       # [s]
OUTPUT_TIME_STEP = 0.02          # [s]
SOLVER_MAX_STEP = 0.05           # [s]
PHASE_POINTS = 721               # 1-degree points over a 720-degree cycle

# A case is treated as settled when every later complete cycle remains
# sufficiently close to the last complete cycle.
SETTLING_MEAN_TOL = 0.01         # 1% tolerance on cycle mean speed
SETTLING_RIPPLE_TOL = 0.03       # 3% tolerance on cycle speed variation
MIN_SETTLED_CYCLES = 5

MAKE_ANIMATION = True          # Change to True to show selected-case animation
ANIMATION_DURATION = 8.0         # [s]
ANIMATION_FRAME_SKIP = 4

# ============================================================
# 2. MECHANISM PARAMETERS
# ============================================================
m1 = 1.0                         # crank mass [kg]
m2 = 2.0                         # connecting rod mass [kg]
m3 = 0.5                         # piston mass [kg]

L1 = 0.5                         # crank length [m]
L2 = 1.5                         # connecting rod length [m]

I_crank = 0.5                    # crank rotational inertia [kg m^2]
I2 = 1.5                         # connecting rod rotational inertia [kg m^2]

g = 9.81                         # gravitational acceleration [m/s^2]

# Increased from 0.05 so that the loaded engine approaches repeatable
# operation within a reasonable simulation duration.
c_rot = 0.10                     # resisting torque coefficient [N m s/rad]

# ============================================================
# 3. FLYWHEEL CASES
# ============================================================
NO_FLYWHEEL = {
    'label': 'No flywheel',
    'mass': 0.0,
    'radius': 0.0,
    'inertia': 0.0,
}

SELECTED_FLYWHEEL = {
    'label': 'Selected flywheel',
    'mass': 12.5,                # [kg]
    'radius': 0.40,              # [m]
}
SELECTED_FLYWHEEL['inertia'] = (
    0.5 * SELECTED_FLYWHEEL['mass'] * SELECTED_FLYWHEEL['radius']**2
)

COMPARISON_CASES = [NO_FLYWHEEL, SELECTED_FLYWHEEL]

# ============================================================
# 4. OTTO-CYCLE PARAMETERS
# ============================================================
p_atm = 101325.0                 # atmospheric pressure [Pa]
gamma = 1.4                      # heat-capacity ratio
compression_ratio = 8.0          # V_BDC / V_TDC
A_piston = 2.0e-5                # piston area [m^2]

stroke = 2 * L1                  # [m]
V_displaced = A_piston * stroke
V_clearance = V_displaced / (compression_ratio - 1)
V_TDC = V_clearance
V_BDC = V_clearance + V_displaced
y_TDC = L1 + L2

p_peak = 4.0e6                   # peak pressure after ignition [Pa]
p_compression_TDC = p_atm * compression_ratio**gamma

print(f'Compression pressure at TDC: {p_compression_TDC / 1e6:.3f} MPa')
print(f'Combustion peak pressure:     {p_peak / 1e6:.3f} MPa')
print(f"Selected flywheel inertia:    {SELECTED_FLYWHEEL['inertia']:.4f} kg m^2")

# ============================================================
# 5. SYMBOLIC MULTIBODY MODEL
# ============================================================
t = sp.symbols('t')
x1, y1, theta1, x2, y2, theta2, x3, y3 = dynamicsymbols(
    'x1 y1 theta1 x2 y2 theta2 x3 y3'
)

q_sym = sp.Matrix([x1, y1, theta1, x2, y2, theta2, x3, y3])
dq_sym = q_sym.diff(t)


def R(theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta),  sp.cos(theta)]
    ])


r1 = sp.Matrix([x1, y1])
r2 = sp.Matrix([x2, y2])
r3 = sp.Matrix([x3, y3])

# Constraint equations: fixed crank pivot, crank/rod pin, rod/piston pin,
# and vertical piston guide.
fixed_pivot = r1 + R(theta1) @ sp.Matrix([-L1 / 2, 0])
crank_rod_joint = (
    r1 + R(theta1) @ sp.Matrix([L1 / 2, 0])
    - r2 - R(theta2) @ sp.Matrix([-L2 / 2, 0])
)
rod_piston_joint = r2 + R(theta2) @ sp.Matrix([L2 / 2, 0]) - r3

C = sp.Matrix([
    fixed_pivot[0], fixed_pivot[1],
    crank_rod_joint[0], crank_rod_joint[1],
    rod_piston_joint[0], rod_piston_joint[1],
    r3[0]
])

J = C.jacobian(q_sym)
dC = J @ dq_sym
dJ = dC.jacobian(q_sym)

# Gravity and viscous/load resistance; gas force is added numerically.
Q_base = sp.Matrix([
    0, -m1 * g, -c_rot * theta1.diff(t),
    0, -m2 * g, 0,
    0, -m3 * g
])

C_fn = sp.lambdify((q_sym, dq_sym), C, 'numpy')
J_fn = sp.lambdify((q_sym, dq_sym), J, 'numpy')
dC_fn = sp.lambdify((q_sym, dq_sym), dC, 'numpy')
dJ_fn = sp.lambdify((q_sym, dq_sym), dJ, 'numpy')
Q_base_fn = sp.lambdify((q_sym, dq_sym), Q_base, 'numpy')

# ============================================================
# 6. OTTO-CYCLE FORCE AND GAS TORQUE
# ============================================================
def cycle_angle(theta):
    """Four-stroke phase from TDC in the range 0 <= phi < 4*pi."""
    return np.mod(theta - np.pi / 2, 4 * np.pi)


def stroke_name(theta):
    phi = cycle_angle(theta)
    if phi < np.pi:
        return 'Power'
    if phi < 2 * np.pi:
        return 'Exhaust'
    if phi < 3 * np.pi:
        return 'Intake'
    return 'Compression'


def cylinder_pressure(theta, y_piston):
    phi = cycle_angle(theta)
    V = V_clearance + A_piston * (y_TDC - y_piston)
    V = np.clip(V, V_TDC, V_BDC)

    if phi < np.pi:
        return p_peak * (V_TDC / V)**gamma
    if phi < 2 * np.pi:
        return p_atm
    if phi < 3 * np.pi:
        return p_atm
    return p_atm * (V_BDC / V)**gamma


def otto_piston_force(theta, y_piston):
    """Vertical piston gas force [N], with upward positive."""
    return -(cylinder_pressure(theta, y_piston) - p_atm) * A_piston


def dy3_dtheta1(theta):
    """Slider velocity ratio for the upper assembly branch."""
    root = np.sqrt(np.maximum(L2**2 - (L1 * np.cos(theta))**2, 1e-14))
    return L1 * np.cos(theta) + (
        L1**2 * np.sin(theta) * np.cos(theta) / root
    )


def gas_torque(theta, y_piston):
    """Generalised gas torque about crank angle theta1 [N m]."""
    return otto_piston_force(theta, y_piston) * dy3_dtheta1(theta)

# ============================================================
# 7. INITIAL CONDITIONS
# ============================================================
def consistent_initial_state(omega0=4.0):
    q0 = np.array([
        0.0, L1 / 2, np.pi / 2,
        0.0, L1 + L2 / 2, np.pi / 2,
        0.0, L1 + L2
    ])

    J0 = np.asarray(J_fn(q0, np.zeros(8)), dtype=float)
    dq0 = np.zeros(8)
    dq0[2] = omega0
    unknown = [0, 1, 3, 4, 5, 6, 7]
    dq0[unknown] = np.linalg.solve(J0[:, unknown], -J0[:, 2] * omega0)

    if not np.allclose(np.asarray(C_fn(q0, dq0), dtype=float), 0.0, atol=1e-10):
        raise ValueError('Initial position constraints are not satisfied.')
    if not np.allclose(np.asarray(dC_fn(q0, dq0), dtype=float), 0.0, atol=1e-10):
        raise ValueError('Initial velocity constraints are not satisfied.')

    return np.concatenate((q0, dq0))


omega0 = 4.0
x0 = consistent_initial_state(omega0)

baumgarte_frequency = 20.0
baumgarte_damping_ratio = 1.0

# ============================================================
# 8. CASE-SPECIFIC DYNAMICS AND SOLUTION
# ============================================================
def build_dynamics(I_flywheel):
    I_total = I_crank + I_flywheel
    M = np.diag([m1, m1, I_total, m2, m2, I2, m3, m3])
    W = np.linalg.inv(M)

    def evaluate(time, state):
        q, dq = np.split(state, 2)
        J_num = np.asarray(J_fn(q, dq), dtype=float)
        dJ_num = np.asarray(dJ_fn(q, dq), dtype=float)
        C_num = np.asarray(C_fn(q, dq), dtype=float).reshape(-1, 1)
        dC_num = np.asarray(dC_fn(q, dq), dtype=float).reshape(-1, 1)

        Q = np.asarray(Q_base_fn(q, dq), dtype=float).reshape(-1, 1)
        F_gas = otto_piston_force(q[2], q[7])
        Q[7, 0] += F_gas

        JWJT = J_num @ W @ J_num.T
        rhs_lambda = (
            -dJ_num @ dq.reshape(-1, 1)
            -J_num @ W @ Q
            -(baumgarte_frequency**2) * C_num
            -2 * baumgarte_damping_ratio * baumgarte_frequency * dC_num
        )
        lam = np.linalg.solve(JWJT, rhs_lambda)
        ddq = W @ (Q + J_num.T @ lam)
        derivative = np.concatenate((dq, ddq.flatten()))
        return derivative, lam.flatten()

    def rhs(time, state):
        return evaluate(time, state)[0]

    return rhs, evaluate


def run_case(case):
    rhs, evaluate = build_dynamics(case['inertia'])
    t_eval = np.arange(0.0, SIMULATION_END_TIME + OUTPUT_TIME_STEP / 2,
                       OUTPUT_TIME_STEP)
    sol = solve_ivp(
        rhs, (0.0, SIMULATION_END_TIME), x0,
        method='BDF', t_eval=t_eval,
        rtol=1e-7, atol=1e-9, max_step=SOLVER_MAX_STEP
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed for {case['label']}: {sol.message}")

    theta = sol.y[2]
    y_piston = sol.y[7]
    omega = sol.y[10]
    alpha = np.zeros_like(sol.t)
    reaction_A = np.zeros_like(sol.t)
    reaction_B = np.zeros_like(sol.t)
    error = np.zeros_like(sol.t)

    for i in range(len(sol.t)):
        derivative, lam = evaluate(sol.t[i], sol.y[:, i])
        alpha[i] = derivative[10]
        reaction_A[i] = np.hypot(lam[2], lam[3])
        reaction_B[i] = np.hypot(lam[4], lam[5])
        error[i] = np.max(np.abs(np.asarray(
            C_fn(sol.y[:8, i], sol.y[8:, i]), dtype=float
        )))

    F_gas = np.array([otto_piston_force(a, b) for a, b in zip(theta, y_piston)])
    tau_gas = np.array([gas_torque(a, b) for a, b in zip(theta, y_piston)])
    tau_resistance = -c_rot * omega
    rotational_energy = 0.5 * (I_crank + case['inertia']) * omega**2

    return {
        **case,
        'sol': sol, 'theta': theta, 'y_piston': y_piston,
        'omega': omega, 'alpha': alpha, 'F_gas': F_gas,
        'tau_gas': tau_gas, 'tau_resistance': tau_resistance,
        'rotational_energy': rotational_energy,
        'reaction_A': reaction_A, 'reaction_B': reaction_B,
        'constraint_error': error
    }

# ============================================================
# 9. PHASE-INTERPOLATED COMPLETE CYCLE METRICS
# ============================================================
def get_phase_cycle(data, cycle_number):
    """Interpolate one complete 720-degree cycle on a common phase grid."""
    theta0 = np.pi / 2 + cycle_number * 4 * np.pi
    theta_grid = theta0 + np.linspace(0.0, 4 * np.pi, PHASE_POINTS)
    phase_deg = np.linspace(0.0, 720.0, PHASE_POINTS)

    if theta_grid[-1] > data['theta'][-1]:
        raise ValueError('Requested cycle is incomplete.')

    interp = {'phase_deg': phase_deg, 'theta': theta_grid}
    for key in ['omega', 'alpha', 'F_gas', 'tau_gas', 'tau_resistance',
                'rotational_energy', 'reaction_A', 'reaction_B',
                'constraint_error']:
        interp[key] = np.interp(theta_grid, data['theta'], data[key])
    interp['time'] = np.interp(theta_grid, data['theta'], data['sol'].t)
    return interp


def complete_cycle_numbers(data):
    total_rotation = data['theta'][-1] - np.pi / 2
    n_complete = int(np.floor(total_rotation / (4 * np.pi)))
    return list(range(n_complete))


def metrics_for_cycle(data, cycle_number):
    cyc = get_phase_cycle(data, cycle_number)
    duration = cyc['time'][-1] - cyc['time'][0]
    omega_max = np.max(cyc['omega'])
    omega_min = np.min(cyc['omega'])
    omega_mean = 4 * np.pi / duration
    delta_omega = omega_max - omega_min
    return {
        'cycle_number': cycle_number,
        'cycle': cyc,
        'start_time': cyc['time'][0],
        'end_time': cyc['time'][-1],
        'omega_max': omega_max,
        'omega_min': omega_min,
        'omega_mean': omega_mean,
        'delta_omega': delta_omega,
        'Cs': delta_omega / omega_mean,
        'alpha_max_abs': np.max(np.abs(cyc['alpha'])),
        'F_gas_max_abs': np.max(np.abs(cyc['F_gas'])),
        'tau_gas_max_abs': np.max(np.abs(cyc['tau_gas'])),
        'tau_resistance_max_abs': np.max(np.abs(cyc['tau_resistance'])),
        'rot_energy_variation': np.max(cyc['rotational_energy']) - np.min(cyc['rotational_energy']),
        'reaction_A_max': np.max(cyc['reaction_A']),
        'reaction_B_max': np.max(cyc['reaction_B']),
        'constraint_error_max': np.max(cyc['constraint_error']),
    }


def analyse_case(data):
    peak_index = int(np.argmax(data['omega']))
    data['peak_time_recorded'] = data['sol'].t[peak_index]
    data['peak_speed_recorded'] = data['omega'][peak_index]

    cycle_numbers = complete_cycle_numbers(data)
    if len(cycle_numbers) < MIN_SETTLED_CYCLES + 1:
        raise RuntimeError(f"Too few complete cycles for {data['label']}.")

    all_metrics = [metrics_for_cycle(data, n) for n in cycle_numbers]
    final_metrics = all_metrics[-1]
    final_mean = final_metrics['omega_mean']
    final_ripple = final_metrics['delta_omega']

    settling_metric = None
    for i in range(len(all_metrics) - MIN_SETTLED_CYCLES + 1):
        later = all_metrics[i:]
        mean_ok = all(
            abs(m['omega_mean'] - final_mean) / abs(final_mean) <= SETTLING_MEAN_TOL
            for m in later
        )
        ripple_ok = all(
            abs(m['delta_omega'] - final_ripple) / abs(final_ripple) <= SETTLING_RIPPLE_TOL
            for m in later
        )
        if mean_ok and ripple_ok:
            settling_metric = all_metrics[i]
            break

    data['all_cycle_metrics'] = all_metrics
    data['final_cycle'] = final_metrics
    data['settled'] = settling_metric is not None
    data['settling_time'] = settling_metric['start_time'] if settling_metric else np.nan
    return data

# ============================================================
# 10. RUN CASES
# ============================================================
results = {}
for case in COMPARISON_CASES:
    print(f"\nSolving: {case['label']} ...")
    results[case['label']] = analyse_case(run_case(case))

no = results['No flywheel']
fly = results['Selected flywheel']
N = no['final_cycle']
F = fly['final_cycle']


def reduction(no_value, fly_value):
    return (no_value - fly_value) / no_value * 100

# ============================================================
# 11. PRINT RESULTS
# ============================================================
rows = [
    ('Time of largest recorded speed [s]', no['peak_time_recorded'], fly['peak_time_recorded']),
    ('Largest recorded speed [rad/s]', no['peak_speed_recorded'], fly['peak_speed_recorded']),
    ('Estimated settling time [s]', no['settling_time'], fly['settling_time']),
    ('Final cycle start time [s]', N['start_time'], F['start_time']),
    ('Final cycle maximum speed [rad/s]', N['omega_max'], F['omega_max']),
    ('Final cycle minimum speed [rad/s]', N['omega_min'], F['omega_min']),
    ('Final cycle mean speed [rad/s]', N['omega_mean'], F['omega_mean']),
    ('Final cycle speed variation [rad/s]', N['delta_omega'], F['delta_omega']),
    ('Coefficient of speed fluctuation C_s', N['Cs'], F['Cs']),
    ('Maximum |angular acceleration| [rad/s^2]', N['alpha_max_abs'], F['alpha_max_abs']),
    ('Maximum |gas force| [N]', N['F_gas_max_abs'], F['F_gas_max_abs']),
    ('Maximum |gas torque| [N m]', N['tau_gas_max_abs'], F['tau_gas_max_abs']),
    ('Maximum |resisting torque| [N m]', N['tau_resistance_max_abs'], F['tau_resistance_max_abs']),
    ('Rotational energy variation [J]', N['rot_energy_variation'], F['rot_energy_variation']),
    ('Maximum joint A reaction [N]', N['reaction_A_max'], F['reaction_A_max']),
    ('Maximum joint B reaction [N]', N['reaction_B_max'], F['reaction_B_max']),
    ('Maximum constraint error', N['constraint_error_max'], F['constraint_error_max']),
]

print('\n' + '=' * 106)
print('NO-FLYWHEEL VS SELECTED-FLYWHEEL COMPARISON: FINAL COMPLETE OTTO CYCLE')
print('=' * 106)
print(f"Flywheel: m = {fly['mass']:.2f} kg, R = {fly['radius']:.2f} m, I_f = {fly['inertia']:.4f} kg m^2")
print(f"Settled check passed: no flywheel = {no['settled']}, flywheel = {fly['settled']}")
print('-' * 106)
print(f"{'Measure':<52}{'No flywheel':>20}{'Flywheel':>20}")
print('-' * 106)
for label, a, b in rows:
    print(f'{label:<52}{a:>20.6f}{b:>20.6f}')
print('-' * 106)
print(f"Reduction in final-cycle speed variation:        {reduction(N['delta_omega'], F['delta_omega']):.2f}%")
print(f"Reduction in coefficient of speed fluctuation:   {reduction(N['Cs'], F['Cs']):.2f}%")
print(f"Reduction in peak |angular acceleration|:        {reduction(N['alpha_max_abs'], F['alpha_max_abs']):.2f}%")
print(f"Reduction in peak joint A reaction:              {reduction(N['reaction_A_max'], F['reaction_A_max']):.2f}%")
print('=' * 106)

for data in [no, fly]:
    print(f"\nFinal five cycles: {data['label']}")
    print(f"{'Start time [s]':>16}{'Mean speed [rad/s]':>23}{'Variation [rad/s]':>23}")
    for metric in data['all_cycle_metrics'][-5:]:
        print(f"{metric['start_time']:>16.3f}{metric['omega_mean']:>23.5f}{metric['delta_omega']:>23.5f}")

# ============================================================
# 12. SAVE RESULTS TABLE
# ============================================================
with (OUTPUT_DIR / 'flywheel_comparison_results.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Measure', 'No flywheel', 'Selected flywheel'])
    writer.writerows(rows)
    writer.writerow(['Reduction in final-cycle speed variation [%]', '', reduction(N['delta_omega'], F['delta_omega'])])
    writer.writerow(['Reduction in coefficient of speed fluctuation [%]', '', reduction(N['Cs'], F['Cs'])])
    writer.writerow(['Reduction in peak |angular acceleration| [%]', '', reduction(N['alpha_max_abs'], F['alpha_max_abs'])])
    writer.writerow(['Reduction in peak joint A reaction [%]', '', reduction(N['reaction_A_max'], F['reaction_A_max'])])

# ============================================================
# 13. REPORT-READY PLOTS
# ============================================================
def save_show(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.show()

# Full speed history and time of largest recorded speed.
plt.figure(figsize=(11, 5))
for data in [no, fly]:
    plt.plot(data['sol'].t, data['omega'], label=data['label'])
    plt.plot(data['peak_time_recorded'], data['peak_speed_recorded'], marker='o')
    plt.annotate(
        f"t={data['peak_time_recorded']:.2f} s\n$\\omega$={data['peak_speed_recorded']:.2f} rad/s",
        (data['peak_time_recorded'], data['peak_speed_recorded']),
        textcoords='offset points', xytext=(8, 10), fontsize=8
    )
plt.xlabel('Time [s]')
plt.ylabel(r'Crank angular velocity $\dot{\theta}_1$ [rad/s]')
plt.title('Crankshaft Angular Speed and Largest Recorded Speed')
plt.grid()
plt.legend()
save_show('01_full_speed_and_largest_recorded_speed.png')

# Settled/latest full-cycle speed comparison.
plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['omega'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['omega'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel(r'Angular velocity $\dot{\theta}_1$ [rad/s]')
plt.title('Final Complete-Cycle Crankshaft Speed Comparison')
plt.grid()
plt.legend()
save_show('02_final_cycle_speed_comparison.png')

# Ripple about each cycle mean.
plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['omega'] - N['omega_mean'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['omega'] - F['omega_mean'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel(r'$\omega - \overline{\omega}$ [rad/s]')
plt.title('Flywheel Smoothing of Angular-Speed Fluctuation')
plt.grid()
plt.legend()
save_show('03_final_cycle_speed_ripple.png')

# Acceleration comparison.
plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['alpha'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['alpha'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel(r'Angular acceleration $\ddot{\theta}_1$ [rad/s$^2$]')
plt.title('Flywheel Effect on Angular Acceleration')
plt.grid()
plt.legend()
save_show('04_final_cycle_angular_acceleration.png')

# Otto gas torque: should be similar, because flywheel changes response not input law.
plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['tau_gas'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['tau_gas'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel('Gas torque [N m]')
plt.title('Gas Torque Input Over a Complete Otto Cycle')
plt.grid()
plt.legend()
save_show('05_final_cycle_gas_torque.png')

# Rotational energy.
plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['rotational_energy'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['rotational_energy'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel('Rotational kinetic energy [J]')
plt.title('Rotational Energy Storage Over a Complete Otto Cycle')
plt.grid()
plt.legend()
save_show('06_final_cycle_rotational_energy.png')

# Constraint reaction comparisons.
plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['reaction_A'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['reaction_A'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel('Joint A reaction magnitude [N]')
plt.title('Crank-to-Rod Reaction Force Comparison')
plt.grid()
plt.legend()
save_show('07_final_cycle_joint_A_reaction.png')

plt.figure(figsize=(10, 5))
plt.plot(N['cycle']['phase_deg'], N['cycle']['reaction_B'], label='No flywheel')
plt.plot(F['cycle']['phase_deg'], F['cycle']['reaction_B'], label='Selected flywheel')
plt.xlabel('Crank rotation through complete Otto cycle [degrees]')
plt.ylabel('Joint B reaction magnitude [N]')
plt.title('Rod-to-Piston Reaction Force Comparison')
plt.grid()
plt.legend()
save_show('08_final_cycle_joint_B_reaction.png')

# Numerical validity.
plt.figure(figsize=(10, 5))
for data in [no, fly]:
    plt.semilogy(data['sol'].t, np.maximum(data['constraint_error'], 1e-16), label=data['label'])
plt.xlabel('Time [s]')
plt.ylabel('Maximum absolute constraint error')
plt.title('Constraint Error Throughout Simulations')
plt.grid()
plt.legend()
save_show('09_constraint_error.png')

# ============================================================
# 14. OPTIONAL ANIMATION OF SELECTED FLYWHEEL CASE
# ============================================================
class Box:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.offset = -np.array([width / 2, height / 2])
        self.patch = None

    def first_draw(self, ax):
        self.patch = plt.Rectangle((0, 0), self.width, self.height,
                                   angle=0, rotation_point='center', animated=True)
        ax.add_patch(self.patch)
        return self.patch

    def update(self, x, y, theta):
        self.patch.set_xy(np.array([x, y]) + self.offset)
        self.patch.set_angle(np.rad2deg(theta))
        return self.patch


if MAKE_ANIMATION:
    indices = np.where(fly['sol'].t <= ANIMATION_DURATION)[0][::ANIMATION_FRAME_SKIP]
    fig, ax = plt.subplots(figsize=(5, 8))
    ax.set_ylim(-0.6, 2.2)
    ax.set_xlim(-0.7, 0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid()
    ax.plot(0, 0, 'ko', markersize=5)
    ax.plot([-0.08, -0.08], [1.35, 2.15], linestyle='--')
    ax.plot([0.08, 0.08], [1.35, 2.15], linestyle='--')
    ax.add_patch(plt.Circle((0, 0), fly['radius'], fill=False, linewidth=2))
    spoke, = ax.plot([], [], linewidth=2)
    info = ax.text(0.03, 0.95, '', transform=ax.transAxes, verticalalignment='top')
    crank, rod, piston = Box(L1, 0.025), Box(L2, 0.025), Box(0.14, 0.28)
    boxes = [crank, rod, piston]

    def init_animation():
        for box in boxes:
            box.first_draw(ax)
        return [box.patch for box in boxes] + [spoke, info]

    def animate(frame):
        i = indices[frame]
        state = fly['sol'].y[:, i]
        theta = fly['theta'][i]
        crank.update(state[0], state[1], state[2])
        rod.update(state[3], state[4], state[5])
        piston.update(state[6], state[7], 0.0)
        spoke.set_data([0, fly['radius'] * np.cos(theta)],
                       [0, fly['radius'] * np.sin(theta)])
        info.set_text(f"t = {fly['sol'].t[i]:.2f} s\nStroke: {stroke_name(theta)}\n"
                      f"Speed: {fly['omega'][i]:.2f} rad/s")
        ax.set_title('Selected Flywheel Slider-Crank Motion')
        return [box.patch for box in boxes] + [spoke, info]

    animation = FuncAnimation(fig, animate, frames=len(indices), init_func=init_animation,
                              interval=1000 * OUTPUT_TIME_STEP * ANIMATION_FRAME_SKIP,
                              blit=False)
    plt.show()

print(f'\nSaved plots and CSV file in: {OUTPUT_DIR.resolve()}')
