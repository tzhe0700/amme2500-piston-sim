# ============================================================
# FOUR-STROKE OTTO CYCLE PISTON ENGINE WITH FLYWHEEL
# Constrained multibody dynamics model
# ============================================================

# -----------------------------
# 1. Import libraries
# -----------------------------
import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display

from scipy.integrate import solve_ivp


# -----------------------------
# 2. Physical parameters
# -----------------------------

# Link / piston masses [kg]
m1 = 1.0       # crank mass
m2 = 2.0       # connecting rod mass
m3 = 0.5       # piston mass

# Link lengths [m]
L1 = 0.5       # crank length
L2 = 1.5       # connecting rod length

# Original rotational inertias [kg m^2]
I_crank = 0.5
I2 = 1.5

# Gravity [m/s^2]
g = 9.81

# Rotational resistance/load torque coefficient [N m s/rad]
# Torque = -c_rot * angular_velocity
c_rot = 0.05


# -----------------------------
# 3. Flywheel parameters
# -----------------------------

# Set m_flywheel = 0.0 to simulate without a flywheel
m_flywheel = 12.5     # [kg] selected demonstration flywheel
R_flywheel = 0.40      # [m] selected demonstration flywheel radius

# Solid disk flywheel inertia
I_flywheel = 0.5 * m_flywheel * R_flywheel**2

# Crank and flywheel rotate together
I1 = I_crank + I_flywheel

print(f"Flywheel inertia:       {I_flywheel:.4f} kg m^2")
print(f"Total crank inertia:    {I1:.4f} kg m^2")


# -----------------------------
# 4. Otto-cycle cylinder parameters
# -----------------------------

p_atm = 101325.0            # atmospheric pressure [Pa]
gamma = 1.4                 # ratio of specific heats for air
compression_ratio = 8.0     # V_BDC / V_TDC

# Small effective piston area because the mechanism dimensions
# in this example are much larger than a real small engine.
A_piston = 2.0e-5           # [m^2]

# Slider-crank stroke
stroke = 2 * L1             # [m]

# Cylinder volumes
V_displaced = A_piston * stroke
V_clearance = V_displaced / (compression_ratio - 1)

V_TDC = V_clearance
V_BDC = V_clearance + V_displaced

# Piston vertical position at top dead centre
y_TDC = L1 + L2

# Pressure immediately after ignition at TDC [Pa]
# This must exceed the compression pressure at TDC.
p_peak = 4.0e6

p_compression_TDC = p_atm * compression_ratio**gamma

print(f"Compression pressure at TDC: {p_compression_TDC / 1e6:.3f} MPa")
print(f"Combustion peak pressure:     {p_peak / 1e6:.3f} MPa")


# -----------------------------
# 5. Mass matrix
# -----------------------------

M = np.diag([
    m1, m1, I1,       # body 1: x1, y1, theta1
    m2, m2, I2,       # body 2: x2, y2, theta2
    m3, m3            # body 3: x3, y3
])

W = np.linalg.inv(M)


# -----------------------------
# 6. Symbolic generalised coordinates
# -----------------------------

t = sp.symbols('t')

x1, y1, theta1, x2, y2, theta2, x3, y3 = dynamicsymbols(
    'x1 y1 theta1 x2 y2 theta2 x3 y3'
)

q_sym = sp.Matrix([
    x1, y1, theta1,
    x2, y2, theta2,
    x3, y3
])

dq_sym = q_sym.diff(t)


# -----------------------------
# 7. Rotation matrix
# -----------------------------

def R(theta):
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)],
        [sp.sin(theta),  sp.cos(theta)]
    ])


i_cap = sp.Matrix([1, 0])
j_cap = sp.Matrix([0, 1])

x_com_1 = sp.Matrix([x1, y1])
x_com_2 = sp.Matrix([x2, y2])
x_com_3 = sp.Matrix([x3, y3])


# -----------------------------
# 8. Constraint equations
# -----------------------------

# Body 1 left end fixed at the origin
constraint_1 = x_com_1 + R(theta1) @ sp.Matrix([-L1 / 2, 0])

C1 = constraint_1.dot(i_cap)
C2 = constraint_1.dot(j_cap)

# Body 1 right end connected to body 2 left end
constraint_2 = (
    x_com_1
    - x_com_2
    + R(theta1) @ sp.Matrix([L1 / 2, 0])
    - R(theta2) @ sp.Matrix([-L2 / 2, 0])
)

C3 = constraint_2.dot(i_cap)
C4 = constraint_2.dot(j_cap)

# Body 2 right end connected to piston
constraint_3 = (
    x_com_2
    + R(theta2) @ sp.Matrix([L2 / 2, 0])
    - x_com_3
)

C5 = constraint_3.dot(i_cap)
C6 = constraint_3.dot(j_cap)

# Piston constrained to move vertically only
C7 = x_com_3[0]

C = sp.Matrix([C1, C2, C3, C4, C5, C6, C7])


# -----------------------------
# 9. Constraint Jacobian terms
# -----------------------------

J = C.jacobian(q_sym)

dC = J @ dq_sym

# For constraints with no explicit time dependence:
# d/dt(J dq) = J ddq + dJ dq
dJ = dC.jacobian(q_sym)


# -----------------------------
# 10. Base generalised forces
# -----------------------------

# The combustion/piston gas force is added numerically inside the ODE.
# Only gravity and crank resistance are included symbolically here.

Q_base = sp.Matrix([
    0,
    -m1 * g,
    -c_rot * theta1.diff(t),

    0,
    -m2 * g,
    0,

    0,
    -m3 * g
])


# -----------------------------
# 11. Lambdified symbolic functions
# -----------------------------

JWJT = J @ sp.Matrix(W) @ J.T

C_fn = sp.lambdify((q_sym, dq_sym), C, 'numpy')
J_fn = sp.lambdify((q_sym, dq_sym), J, 'numpy')
dC_fn = sp.lambdify((q_sym, dq_sym), dC, 'numpy')
dJ_fn = sp.lambdify((q_sym, dq_sym), dJ, 'numpy')
Q_base_fn = sp.lambdify((q_sym, dq_sym), Q_base, 'numpy')
JWJT_fn = sp.lambdify((q_sym, dq_sym), JWJT, 'numpy')


# -----------------------------
# 12. Four-stroke Otto-cycle force
# -----------------------------

def otto_piston_force(theta, y_piston):
    """
    Return the vertical gas force acting on the piston.

    Model convention:
        theta = pi/2 is top dead centre.
        Positive angular velocity begins the power stroke.
        Positive y is upward.
        Combustion pressure pushes the piston downward, so the
        returned gas force is negative during the power stroke.

    Stroke ranges:
        0      <= phi < pi      : power / expansion
        pi     <= phi < 2*pi    : exhaust
        2*pi   <= phi < 3*pi    : intake
        3*pi   <= phi < 4*pi    : compression
    """

    # Crank-cycle angle measured from TDC.
    # Four-stroke cycle repeats every 4*pi rad.
    phi = np.mod(theta - np.pi / 2, 4 * np.pi)

    # Cylinder volume based on piston displacement below TDC
    V = V_clearance + A_piston * (y_TDC - y_piston)

    # Prevent small numerical drift from creating non-physical volume
    V = np.clip(V, V_TDC, V_BDC)

    if phi < np.pi:
        # Power stroke: combustion followed by adiabatic expansion
        p_cylinder = p_peak * (V_TDC / V)**gamma

    elif phi < 2 * np.pi:
        # Exhaust stroke: approximate cylinder pressure as atmospheric
        p_cylinder = p_atm

    elif phi < 3 * np.pi:
        # Intake stroke: approximate cylinder pressure as atmospheric
        p_cylinder = p_atm

    else:
        # Compression stroke: adiabatic compression from atmospheric pressure
        p_cylinder = p_atm * (V_BDC / V)**gamma

    # Gauge pressure produces piston force.
    # Negative sign: gas pressure pushes piston downward.
    F_gas = -(p_cylinder - p_atm) * A_piston

    return F_gas


def otto_stroke_name(theta):
    """
    Return the current named stroke for plotting/diagnostics.
    """
    phi = np.mod(theta - np.pi / 2, 4 * np.pi)

    if phi < np.pi:
        return "Power"
    elif phi < 2 * np.pi:
        return "Exhaust"
    elif phi < 3 * np.pi:
        return "Intake"
    else:
        return "Compression"


# -----------------------------
# 13. Consistent initial conditions
# -----------------------------

def consistent_initial_state(omega0=4.0):
    """
    Construct initial position and velocity satisfying:
        C(q) = 0
        J(q) dq = 0

    omega0 is the specified initial angular speed of the crank.
    """

    # At theta = pi/2, the piston is at TDC
    q0 = np.array([
        0.0, L1 / 2, np.pi / 2,
        0.0, L1 + L2 / 2, np.pi / 2,
        0.0, L1 + L2
    ])

    dq_trial = np.zeros(8)

    J0 = np.asarray(J_fn(q0, dq_trial), dtype=float)

    # Prescribe crank angular speed dq[2] = omega0
    fixed_index = 2
    unknown_indices = [0, 1, 3, 4, 5, 6, 7]

    dq0 = np.zeros(8)
    dq0[fixed_index] = omega0

    # Solve J dq = 0 for all remaining velocities
    dq0[unknown_indices] = np.linalg.solve(
        J0[:, unknown_indices],
        -J0[:, fixed_index] * omega0
    )

    state0 = np.concatenate((q0, dq0))

    # Check constraints
    C_initial = np.asarray(C_fn(q0, dq0), dtype=float).flatten()
    dC_initial = np.asarray(dC_fn(q0, dq0), dtype=float).flatten()

    print(f"Maximum initial position constraint error: {np.max(np.abs(C_initial)):.3e}")
    print(f"Maximum initial velocity constraint error: {np.max(np.abs(dC_initial)):.3e}")

    assert np.allclose(C_initial, 0.0, atol=1e-10), \
        "Initial position constraint violated"

    assert np.allclose(dC_initial, 0.0, atol=1e-10), \
        "Initial velocity constraint violated"

    return state0


# Initial crank angular speed [rad/s]
omega0 = 4.0

x0 = consistent_initial_state(omega0=omega0)


# -----------------------------
# 14. Constrained equations of motion
# -----------------------------

# Baumgarte constraint stabilisation parameters
baumgarte_frequency = 20.0
baumgarte_damping_ratio = 1.0


def piston_engine(t, state):
    """
    Return the derivative of the constrained engine state.

    State arrangement:
        state = [q, dq]
    """

    q, dq = np.split(state, 2)

    # Evaluate constraint matrices
    J_num = np.asarray(J_fn(q, dq), dtype=float)
    dJ_num = np.asarray(dJ_fn(q, dq), dtype=float)

    C_num = np.asarray(C_fn(q, dq), dtype=float).reshape(-1, 1)
    dC_num = np.asarray(dC_fn(q, dq), dtype=float).reshape(-1, 1)

    dq_col = dq.reshape(-1, 1)

    # Gravity and crank load torque
    Q_num = np.asarray(Q_base_fn(q, dq), dtype=float).reshape(-1, 1)

    # Add four-stroke cylinder force to piston vertical coordinate y3
    F_gas = otto_piston_force(q[2], q[7])
    Q_num[7, 0] += F_gas

    # Constraint-force system:
    #
    # J W J^T lambda =
    #     -dJ dq
    #     -J W Q
    #     -omega_b^2 C
    #     -2 zeta omega_b dC
    #
    JWJT_num = np.asarray(JWJT_fn(q, dq), dtype=float)

    RHS_num = (
        -dJ_num @ dq_col
        -J_num @ W @ Q_num
        -(baumgarte_frequency**2) * C_num
        -2 * baumgarte_damping_ratio * baumgarte_frequency * dC_num
    )

    # Solve for Lagrange multipliers
    lam = np.linalg.solve(JWJT_num, RHS_num)

    # Constraint forces
    Q_constraint = J_num.T @ lam

    # Accelerations
    ddq = W @ (Q_num + Q_constraint)

    return np.concatenate((dq, ddq.flatten()))


# Quick check of derivative at initial state
initial_derivative = piston_engine(0.0, x0)
print(f"Initial state derivative evaluated successfully: {initial_derivative.shape}")


# -----------------------------
# 15. Solve the system
# -----------------------------

t_span = (0.0, 30.0)
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol = solve_ivp(
    piston_engine,
    t_span,
    x0,
    method='BDF',
    t_eval=t_eval,
    rtol=1e-7,
    atol=1e-9,
    max_step=0.02
)

print(f"Solver successful: {sol.success}")
print(sol.message)

if not sol.success:
    raise RuntimeError("The ODE solver failed.")


# -----------------------------
# 16. Extract solution quantities
# -----------------------------

# Positions
x1_sol = sol.y[0]
y1_sol = sol.y[1]
theta1_sol = sol.y[2]

x2_sol = sol.y[3]
y2_sol = sol.y[4]
theta2_sol = sol.y[5]

x3_sol = sol.y[6]
y3_sol = sol.y[7]

# Velocities
omega1_sol = sol.y[10]       # dq coordinate corresponding to theta1

# Force history
F_gas_sol = np.array([
    otto_piston_force(theta, y_piston)
    for theta, y_piston in zip(theta1_sol, y3_sol)
])

# Stroke phase labels
stroke_sol = np.array([
    otto_stroke_name(theta)
    for theta in theta1_sol
])

# Constraint-error history
constraint_error = np.array([
    np.max(np.abs(np.asarray(C_fn(sol.y[:8, i], sol.y[8:, i]), dtype=float)))
    for i in range(len(sol.t))
])

print(f"Maximum position constraint error: {np.max(constraint_error):.3e}")
print(f"Angular speed variation: {np.max(omega1_sol) - np.min(omega1_sol):.4f} rad/s")


# -----------------------------
# 17. Diagnostic plots
# -----------------------------

plt.figure(figsize=(9, 4))
plt.plot(sol.t, F_gas_sol)
plt.xlabel('Time [s]')
plt.ylabel('Gas force on piston [N]')
plt.title('Four-Stroke Otto-Cycle Gas Force')
plt.grid()
plt.tight_layout()
plt.savefig('four_stroke_otto_cycle_gas_force.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(9, 4))
plt.plot(sol.t, omega1_sol)
plt.xlabel('Time [s]')
plt.ylabel(r'Crank angular velocity $\dot{\theta}_1$ [rad/s]')
plt.title('Crankshaft Angular Speed for Selected Flywheel')
plt.grid()
plt.tight_layout()
plt.savefig('crankshaft_angular_speed_with_flywheel.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(9, 4))
plt.plot(sol.t, y3_sol)
plt.xlabel('Time [s]')
plt.ylabel('Piston height $y_3$ [m]')
plt.title('Piston Vertical Position')
plt.grid()
plt.tight_layout()
plt.savefig('piston_vertical_position.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(9, 4))
plt.semilogy(sol.t, np.maximum(constraint_error, 1e-16))
plt.xlabel('Time [s]')
plt.ylabel('Maximum absolute constraint error')
plt.title('Constraint Error During Simulation')
plt.grid()
plt.tight_layout()
plt.savefig('constraint_error_during_simulation.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 18. Animation drawing classes
# -----------------------------

class Box:
    def __init__(self, width, height, color='b'):
        self.width = width
        self.height = height
        self.color = color
        self.offset = -np.array([width / 2, height / 2])

    def first_draw(self, ax):
        corner = np.array([0.0, 0.0])

        self.patch = plt.Rectangle(
            corner,
            0,
            0,
            angle=0,
            rotation_point='center',
            color=self.color,
            animated=True
        )

        ax.add_patch(self.patch)
        self.ax = ax

        return self.patch

    def set_data(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def update(self, i):
        x = self.x[i]
        y = self.y[i]
        theta_deg = np.rad2deg(self.theta[i])

        corner = np.array([x, y]) + self.offset

        self.patch.set_width(self.width)
        self.patch.set_height(self.height)
        self.patch.set_xy(corner)
        self.patch.set_angle(theta_deg)

        return self.patch


# -----------------------------
# 19. Create engine animation
# -----------------------------

fig, ax = plt.subplots(figsize=(5, 8))

ax.set_ylim(-0.6, 2.2)
ax.set_xlim(-0.7, 0.7)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid()

# Fixed pivot marker
ax.plot(0, 0, 'ko', markersize=5)

# Piston guide rails
ax.plot([-0.08, -0.08], [1.35, 2.15], linestyle='--')
ax.plot([0.08, 0.08], [1.35, 2.15], linestyle='--')

# Flywheel circle
flywheel_circle = plt.Circle(
    (0, 0),
    R_flywheel,
    fill=False,
    linewidth=2
)
ax.add_patch(flywheel_circle)

# Flywheel rotating spoke
flywheel_spoke, = ax.plot([], [], linewidth=2)

# Current stroke text
stroke_text = ax.text(
    0.03,
    0.95,
    '',
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top'
)

# Bodies
box1 = Box(L1, 0.025, 'b')      # crank
box2 = Box(L2, 0.025, 'r')      # connecting rod
box3 = Box(0.14, 0.28, 'g')     # piston

box1.set_data(x1_sol, y1_sol, theta1_sol)
box2.set_data(x2_sol, y2_sol, theta2_sol)
box3.set_data(x3_sol, y3_sol, np.zeros_like(x3_sol))

boxes = [box1, box2, box3]


def init():
    ax.set_title('t = 0.00 s')

    for box in boxes:
        box.first_draw(ax)

    theta = theta1_sol[0]

    flywheel_spoke.set_data(
        [0, R_flywheel * np.cos(theta)],
        [0, R_flywheel * np.sin(theta)]
    )

    stroke_text.set_text(
        f"Stroke: {stroke_sol[0]}\n"
        f"Gas force: {F_gas_sol[0]:.2f} N\n"
        f"Speed: {omega1_sol[0]:.2f} rad/s"
    )

    return [box.patch for box in boxes] + [flywheel_spoke, stroke_text]


def animate(i):
    ax.set_title(f't = {sol.t[i]:.2f} s')

    for box in boxes:
        box.update(i)

    theta = theta1_sol[i]

    flywheel_spoke.set_data(
        [0, R_flywheel * np.cos(theta)],
        [0, R_flywheel * np.sin(theta)]
    )

    stroke_text.set_text(
        f"Stroke: {stroke_sol[i]}\n"
        f"Gas force: {F_gas_sol[i]:.2f} N\n"
        f"Speed: {omega1_sol[i]:.2f} rad/s"
    )

    return [box.patch for box in boxes] + [flywheel_spoke, stroke_text]


frame_interval_ms = 1000 * (sol.t[1] - sol.t[0])

anim = FuncAnimation(
    fig,
    animate,
    frames=len(sol.t),
    init_func=init,
    interval=frame_interval_ms,
    blit=False
)

plt.show()

# ============================================================
# 20. PEAK-SPEED FLYWHEEL COMPARISON STUDY
# ============================================================
# This section compares the no-flywheel system to the selected flywheel
# at the time of maximum simulated speed and over the last COMPLETE
# Otto cycle nearest that peak. Comparing a complete cycle avoids mixing
# startup acceleration with the within-cycle smoothing effect.

comparison_t_span = (0.0, 30.0)
comparison_t_eval = np.linspace(comparison_t_span[0], comparison_t_span[1], 1501)

# Selected visible smoothing case used for the comparison:
# I_f = 1.0 kg m^2 is obtained with a solid disk of mass 12.5 kg,
# radius 0.40 m: I_f = 0.5*m*R^2 = 1.0 kg m^2.
SELECTED_FLYWHEEL_MASS = 12.5       # [kg]
SELECTED_FLYWHEEL_RADIUS = 0.40     # [m]
SELECTED_FLYWHEEL_INERTIA = 0.5 * SELECTED_FLYWHEEL_MASS * SELECTED_FLYWHEEL_RADIUS**2

comparison_cases = {
    'No flywheel': 0.0,
    'Selected flywheel': SELECTED_FLYWHEEL_INERTIA
}


def solve_comparison_case(label, I_f_case):
    """Solve one flywheel case and return histories needed for comparison."""

    I_total_case = I_crank + I_f_case

    M_case = np.diag([
        m1, m1, I_total_case,
        m2, m2, I2,
        m3, m3
    ])
    W_case = np.linalg.inv(M_case)

    def engine_case(t, state, return_details=False):
        q, dq = np.split(state, 2)

        J_num = np.asarray(J_fn(q, dq), dtype=float)
        dJ_num = np.asarray(dJ_fn(q, dq), dtype=float)
        C_num = np.asarray(C_fn(q, dq), dtype=float).reshape(-1, 1)
        dC_num = np.asarray(dC_fn(q, dq), dtype=float).reshape(-1, 1)

        Q_num = np.asarray(Q_base_fn(q, dq), dtype=float).reshape(-1, 1)
        F_gas = otto_piston_force(q[2], q[7])
        Q_num[7, 0] += F_gas

        RHS_num = (
            -dJ_num @ dq.reshape(-1, 1)
            -J_num @ W_case @ Q_num
            -(baumgarte_frequency**2) * C_num
            -2 * baumgarte_damping_ratio * baumgarte_frequency * dC_num
        )

        lam = np.linalg.solve(J_num @ W_case @ J_num.T, RHS_num)
        Q_constraint = J_num.T @ lam
        ddq = W_case @ (Q_num + Q_constraint)
        derivative = np.concatenate((dq, ddq.flatten()))

        if return_details:
            return derivative, lam.flatten(), F_gas
        return derivative

    sol_case = solve_ivp(
        engine_case,
        comparison_t_span,
        x0,
        method='BDF',
        t_eval=comparison_t_eval,
        rtol=1e-7,
        atol=1e-9,
        max_step=0.02
    )

    if not sol_case.success:
        raise RuntimeError(f'Solver failed for case: {label}')

    theta = sol_case.y[2]
    piston_y = sol_case.y[7]
    omega = sol_case.y[10]

    alpha = np.zeros_like(omega)
    force_gas = np.zeros_like(omega)
    reaction_A = np.zeros_like(omega)
    reaction_B = np.zeros_like(omega)
    constraint_error_case = np.zeros_like(omega)

    for i in range(len(sol_case.t)):
        derivative, lam, force_gas[i] = engine_case(
            sol_case.t[i], sol_case.y[:, i], return_details=True
        )
        alpha[i] = derivative[10]

        # Constraint multipliers C3/C4 represent the crank-rod pin pair,
        # and C5/C6 represent the rod-piston pin pair.
        reaction_A[i] = np.hypot(lam[2], lam[3])
        reaction_B[i] = np.hypot(lam[4], lam[5])

        constraint_error_case[i] = np.max(np.abs(
            np.asarray(C_fn(sol_case.y[:8, i], sol_case.y[8:, i]), dtype=float)
        ))

    # Gas torque about the crank angle from virtual work: tau = F_y * dy/dtheta.
    dy_dtheta = np.gradient(piston_y, theta)
    tau_gas = force_gas * dy_dtheta
    tau_resistance = -c_rot * omega

    # Rotational energy stored in the crank/flywheel rotational degree of freedom.
    rotational_energy = 0.5 * I_total_case * omega**2

    # Identify global peak speed and time.
    peak_index = int(np.argmax(omega))
    peak_time = sol_case.t[peak_index]
    peak_speed = omega[peak_index]

    # Find a complete Otto cycle closest to the global peak-speed time.
    # One four-stroke cycle corresponds to 4*pi rad of crank rotation.
    cycle_number = np.floor((theta - np.pi / 2) / (4 * np.pi)).astype(int)
    unique_cycles = np.unique(cycle_number)
    complete_cycles = [
        c for c in unique_cycles
        if c > unique_cycles[0]
        and c < unique_cycles[-1]
        and np.sum(cycle_number == c) >= 5
    ]

    if not complete_cycles:
        raise RuntimeError(f'No complete Otto cycle found for case: {label}')

    peak_cycle_number = cycle_number[peak_index]
    selected_cycle = min(complete_cycles, key=lambda c: abs(c - peak_cycle_number))
    cycle_mask = cycle_number == selected_cycle

    phase_deg = (
        theta[cycle_mask] - (np.pi / 2 + selected_cycle * 4 * np.pi)
    ) * 180 / np.pi

    return {
        'label': label,
        'I_f': I_f_case,
        'I_total': I_total_case,
        'sol': sol_case,
        'theta': theta,
        'omega': omega,
        'alpha': alpha,
        'force_gas': force_gas,
        'tau_gas': tau_gas,
        'tau_resistance': tau_resistance,
        'rotational_energy': rotational_energy,
        'reaction_A': reaction_A,
        'reaction_B': reaction_B,
        'constraint_error': constraint_error_case,
        'peak_index': peak_index,
        'peak_time': peak_time,
        'peak_speed': peak_speed,
        'cycle_number': selected_cycle,
        'cycle_mask': cycle_mask,
        'phase_deg': phase_deg
    }


# -----------------------------
# 20.1 Run no-flywheel and selected-flywheel cases
# -----------------------------

comparison_data = {
    label: solve_comparison_case(label, I_f_case)
    for label, I_f_case in comparison_cases.items()
}


def calculate_cycle_metrics(data):
    """Calculate performance measures over the complete cycle nearest peak speed."""
    m = data['cycle_mask']
    omega_cycle = data['omega'][m]
    alpha_cycle = data['alpha'][m]
    torque_cycle = data['tau_gas'][m]
    resisting_torque_cycle = data['tau_resistance'][m]
    energy_cycle = data['rotational_energy'][m]

    omega_max = np.max(omega_cycle)
    omega_min = np.min(omega_cycle)
    omega_mean = np.mean(omega_cycle)
    delta_omega = omega_max - omega_min
    coefficient_fluctuation = delta_omega / abs(omega_mean)

    return {
        'peak_time': data['peak_time'],
        'global_peak_speed': data['peak_speed'],
        'cycle_start_time': data['sol'].t[m][0],
        'cycle_end_time': data['sol'].t[m][-1],
        'omega_max_cycle': omega_max,
        'omega_min_cycle': omega_min,
        'omega_mean_cycle': omega_mean,
        'delta_omega_cycle': delta_omega,
        'C_s': coefficient_fluctuation,
        'alpha_max_cycle': np.max(alpha_cycle),
        'alpha_min_cycle': np.min(alpha_cycle),
        'alpha_abs_max_cycle': np.max(np.abs(alpha_cycle)),
        'tau_gas_max_cycle': np.max(torque_cycle),
        'tau_gas_min_cycle': np.min(torque_cycle),
        'tau_load_abs_max_cycle': np.max(np.abs(resisting_torque_cycle)),
        'energy_max_cycle': np.max(energy_cycle),
        'energy_min_cycle': np.min(energy_cycle),
        'delta_energy_cycle': np.max(energy_cycle) - np.min(energy_cycle),
        'reaction_A_max_cycle': np.max(data['reaction_A'][m]),
        'reaction_B_max_cycle': np.max(data['reaction_B'][m]),
        'max_constraint_error': np.max(data['constraint_error'])
    }


metrics = {
    label: calculate_cycle_metrics(data)
    for label, data in comparison_data.items()
}

baseline = metrics['No flywheel']
selected = metrics['Selected flywheel']

speed_fluctuation_reduction = (
    (baseline['delta_omega_cycle'] - selected['delta_omega_cycle'])
    / baseline['delta_omega_cycle'] * 100
)

coefficient_reduction = (
    (baseline['C_s'] - selected['C_s'])
    / baseline['C_s'] * 100
)

acceleration_reduction = (
    (baseline['alpha_abs_max_cycle'] - selected['alpha_abs_max_cycle'])
    / baseline['alpha_abs_max_cycle'] * 100
)

reaction_A_change = (
    (selected['reaction_A_max_cycle'] - baseline['reaction_A_max_cycle'])
    / baseline['reaction_A_max_cycle'] * 100
)

reaction_B_change = (
    (selected['reaction_B_max_cycle'] - baseline['reaction_B_max_cycle'])
    / baseline['reaction_B_max_cycle'] * 100
)


# -----------------------------
# 20.2 Print report-ready comparison table
# -----------------------------

print('\n' + '=' * 112)
print('PEAK-SPEED REGION COMPARISON: NO FLYWHEEL VS SELECTED FLYWHEEL')
print('=' * 112)
print(f'Selected flywheel: mass = {SELECTED_FLYWHEEL_MASS:.2f} kg, '
      f'radius = {SELECTED_FLYWHEEL_RADIUS:.2f} m, '
      f'inertia = {SELECTED_FLYWHEEL_INERTIA:.4f} kg m^2')
print('Metrics marked "cycle" are evaluated over the complete Otto cycle nearest the global peak speed.')
print('-' * 112)
print(f"{'Measure':<48}{'No flywheel':>20}{'Selected flywheel':>22}{'Units':>16}")
print('-' * 112)

rows = [
    ('Flywheel inertia', 0.0, SELECTED_FLYWHEEL_INERTIA, 'kg m^2'),
    ('Time of global peak speed', baseline['peak_time'], selected['peak_time'], 's'),
    ('Global peak angular speed', baseline['global_peak_speed'], selected['global_peak_speed'], 'rad/s'),
    ('Cycle start time near peak', baseline['cycle_start_time'], selected['cycle_start_time'], 's'),
    ('Cycle end time near peak', baseline['cycle_end_time'], selected['cycle_end_time'], 's'),
    ('Maximum speed within cycle', baseline['omega_max_cycle'], selected['omega_max_cycle'], 'rad/s'),
    ('Minimum speed within cycle', baseline['omega_min_cycle'], selected['omega_min_cycle'], 'rad/s'),
    ('Mean speed within cycle', baseline['omega_mean_cycle'], selected['omega_mean_cycle'], 'rad/s'),
    ('Speed variation within cycle', baseline['delta_omega_cycle'], selected['delta_omega_cycle'], 'rad/s'),
    ('Coefficient of speed fluctuation', baseline['C_s'], selected['C_s'], '-'),
    ('Maximum absolute angular acceleration', baseline['alpha_abs_max_cycle'], selected['alpha_abs_max_cycle'], 'rad/s^2'),
    ('Maximum gas torque within cycle', baseline['tau_gas_max_cycle'], selected['tau_gas_max_cycle'], 'N m'),
    ('Maximum resisting torque magnitude', baseline['tau_load_abs_max_cycle'], selected['tau_load_abs_max_cycle'], 'N m'),
    ('Rotational energy variation', baseline['delta_energy_cycle'], selected['delta_energy_cycle'], 'J'),
    ('Maximum reaction magnitude at joint A', baseline['reaction_A_max_cycle'], selected['reaction_A_max_cycle'], 'N'),
    ('Maximum reaction magnitude at joint B', baseline['reaction_B_max_cycle'], selected['reaction_B_max_cycle'], 'N'),
    ('Maximum position constraint error', baseline['max_constraint_error'], selected['max_constraint_error'], 'm')
]

for name, no_value, fly_value, units in rows:
    print(f'{name:<48}{no_value:>20.6g}{fly_value:>22.6g}{units:>16}')

print('-' * 112)
print(f'Reduction in cycle speed variation:       {speed_fluctuation_reduction:.2f} %')
print(f'Reduction in coefficient of fluctuation:  {coefficient_reduction:.2f} %')
print(f'Reduction in peak |angular acceleration|: {acceleration_reduction:.2f} %')
print(f'Change in peak joint A reaction:          {reaction_A_change:+.2f} %')
print(f'Change in peak joint B reaction:          {reaction_B_change:+.2f} %')
print('=' * 112)

if baseline['peak_time'] >= comparison_t_span[1] - 1e-9:
    print('Note: The no-flywheel case reaches its largest recorded speed at the end of the simulation.')
    print('Its global peak should therefore be interpreted as the maximum recorded speed, not a confirmed steady maximum.')

if selected['peak_time'] >= comparison_t_span[1] - 1e-9:
    print('Note: The selected-flywheel case reaches its largest recorded speed at the end of the simulation.')
    print('Its global peak should therefore be interpreted as the maximum recorded speed, not a confirmed steady maximum.')


# -----------------------------
# 20.3 Save numerical comparison data
# -----------------------------

summary_array = np.array([
    [0.0, baseline['peak_time'], baseline['global_peak_speed'], baseline['omega_max_cycle'],
     baseline['omega_min_cycle'], baseline['omega_mean_cycle'], baseline['delta_omega_cycle'],
     baseline['C_s'], baseline['alpha_abs_max_cycle'], baseline['tau_gas_max_cycle'],
     baseline['tau_load_abs_max_cycle'], baseline['delta_energy_cycle'],
     baseline['reaction_A_max_cycle'], baseline['reaction_B_max_cycle'], baseline['max_constraint_error']],
    [SELECTED_FLYWHEEL_INERTIA, selected['peak_time'], selected['global_peak_speed'], selected['omega_max_cycle'],
     selected['omega_min_cycle'], selected['omega_mean_cycle'], selected['delta_omega_cycle'],
     selected['C_s'], selected['alpha_abs_max_cycle'], selected['tau_gas_max_cycle'],
     selected['tau_load_abs_max_cycle'], selected['delta_energy_cycle'],
     selected['reaction_A_max_cycle'], selected['reaction_B_max_cycle'], selected['max_constraint_error']]
])

np.savetxt(
    'flywheel_peak_speed_comparison.csv',
    summary_array,
    delimiter=',',
    header=(
        'I_flywheel_kgm2,peak_time_s,global_peak_speed_rads,omega_max_cycle_rads,'
        'omega_min_cycle_rads,omega_mean_cycle_rads,delta_omega_cycle_rads,Cs,'
        'alpha_abs_max_cycle_rads2,tau_gas_max_cycle_Nm,tau_load_abs_max_cycle_Nm,'
        'delta_rotational_energy_cycle_J,reaction_A_max_cycle_N,reaction_B_max_cycle_N,'
        'max_constraint_error_m'
    ),
    comments='',
    fmt='%.8e'
)


# -----------------------------
# 20.4 Full angular-speed history with peak-speed markers
# -----------------------------

plt.figure(figsize=(10, 5))

for label, data in comparison_data.items():
    plt.plot(data['sol'].t, data['omega'], label=label)
    plt.plot(data['peak_time'], data['peak_speed'], marker='o')
    plt.annotate(
        f"{label}\npeak = {data['peak_speed']:.2f} rad/s\nt = {data['peak_time']:.2f} s",
        xy=(data['peak_time'], data['peak_speed']),
        xytext=(-115, -55 if label == 'No flywheel' else 20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->'),
        fontsize=9
    )

plt.xlabel('Time [s]')
plt.ylabel(r'Crank angular velocity $\dot{\theta}_1$ [rad/s]')
plt.title('Crankshaft Angular Speed and Recorded Peak-Speed Time')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('peak_speed_time_flywheel_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 20.5 Speed fluctuation in the complete cycle nearest peak speed
# -----------------------------

plt.figure(figsize=(10, 5))

for label, data in comparison_data.items():
    mask = data['cycle_mask']
    omega_cycle = data['omega'][mask]
    omega_ripple = omega_cycle - np.mean(omega_cycle)
    plt.plot(data['phase_deg'], omega_ripple, label=label)

plt.xlabel(r'Otto-cycle crank angle from TDC [$^\circ$]')
plt.ylabel(r'$\dot{\theta}_1 - \overline{\dot{\theta}_1}$ [rad/s]')
plt.title('Angular-Speed Ripple Over Complete Cycle Nearest Peak Speed')
plt.xlim(0, 720)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('peak_cycle_speed_ripple_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 20.6 Angular acceleration in the complete cycle nearest peak speed
# -----------------------------

plt.figure(figsize=(10, 5))

for label, data in comparison_data.items():
    mask = data['cycle_mask']
    plt.plot(data['phase_deg'], data['alpha'][mask], label=label)

plt.xlabel(r'Otto-cycle crank angle from TDC [$^\circ$]')
plt.ylabel(r'Crank angular acceleration $\ddot{\theta}_1$ [rad/s$^2$]')
plt.title('Angular Acceleration Over Complete Cycle Nearest Peak Speed')
plt.xlim(0, 720)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('peak_cycle_angular_acceleration_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 20.7 Rotational kinetic energy in the complete cycle nearest peak speed
# -----------------------------

plt.figure(figsize=(10, 5))

for label, data in comparison_data.items():
    mask = data['cycle_mask']
    plt.plot(data['phase_deg'], data['rotational_energy'][mask], label=label)

plt.xlabel(r'Otto-cycle crank angle from TDC [$^\circ$]')
plt.ylabel('Crank/flywheel rotational kinetic energy [J]')
plt.title('Rotational Energy Storage Over Complete Cycle Nearest Peak Speed')
plt.xlim(0, 720)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('peak_cycle_rotational_energy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 20.8 Gas torque and resisting torque comparison near peak speed
# -----------------------------

plt.figure(figsize=(10, 5))

for label, data in comparison_data.items():
    mask = data['cycle_mask']
    plt.plot(data['phase_deg'], data['tau_gas'][mask], label=f'{label}: gas torque')

plt.xlabel(r'Otto-cycle crank angle from TDC [$^\circ$]')
plt.ylabel('Gas torque about crank coordinate [N m]')
plt.title('Gas Torque Over Complete Cycle Nearest Peak Speed')
plt.xlim(0, 720)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('peak_cycle_gas_torque_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 20.9 Peak joint reaction comparison
# -----------------------------

labels = ['Joint A\ncrank-rod', 'Joint B\nrod-piston']
no_flywheel_reactions = [baseline['reaction_A_max_cycle'], baseline['reaction_B_max_cycle']]
flywheel_reactions = [selected['reaction_A_max_cycle'], selected['reaction_B_max_cycle']]
bar_positions = np.arange(len(labels))
bar_width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(bar_positions - bar_width / 2, no_flywheel_reactions, bar_width, label='No flywheel')
plt.bar(bar_positions + bar_width / 2, flywheel_reactions, bar_width, label='Selected flywheel')
plt.xticks(bar_positions, labels)
plt.ylabel('Maximum reaction magnitude over selected cycle [N]')
plt.title('Peak Pin-Reaction Comparison Near Peak Speed')
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig('peak_cycle_joint_reaction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------------
# 20.10 Constraint-error validity comparison
# -----------------------------

plt.figure(figsize=(10, 5))
for label, data in comparison_data.items():
    plt.semilogy(
        data['sol'].t,
        np.maximum(data['constraint_error'], 1e-16),
        label=label
    )

plt.xlabel('Time [s]')
plt.ylabel('Maximum absolute constraint error')
plt.title('Numerical Constraint Error for Comparison Cases')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('peak_speed_constraint_error_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
