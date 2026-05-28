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
m_flywheel = 0     # [kg]
R_flywheel = 0.25      # [m]

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
plt.title('Crankshaft Angular Speed with Flywheel')
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
# 20. FLYWHEEL PARAMETRIC RESULTS STUDY
# ============================================================
# Paste this section at the end of the existing code.
#
# This section reruns the model for different flywheel inertias
# and compares crankshaft angular speed and angular acceleration.
#
# For quicker testing, use comparison_t_span = (0.0, 5.0).
# For final report results, change it to (0.0, 30.0).

comparison_t_span = (0.0, 5.0)
comparison_t_eval = np.linspace(
    comparison_t_span[0],
    comparison_t_span[1],
    500
)

# Flywheel inertia values to compare [kg m^2]
# 0.15625 kg m^2 is your existing 5 kg, 0.25 m radius flywheel.
I_flywheel_cases = np.array([
    0.0,
    0.15625,
    0.50,
    1.00,
    2.00
])


def solve_flywheel_case(I_f_case):
    """
    Solve the slider-crank model for one selected flywheel inertia.

    Parameters
    ----------
    I_f_case : float
        Flywheel rotational inertia [kg m^2].

    Returns
    -------
    sol_case : OdeResult
        Numerical solution returned by solve_ivp.
    omega_case : ndarray
        Crankshaft angular velocity history [rad/s].
    alpha_case : ndarray
        Crankshaft angular acceleration history [rad/s^2].
    constraint_case : ndarray
        Maximum position constraint error at each timestep.
    """

    # Total crankshaft rotational inertia for this case
    I_total_case = I_crank + I_f_case

    # Rebuild mass matrix for this flywheel inertia
    M_case = np.diag([
        m1, m1, I_total_case,     # crank and flywheel
        m2, m2, I2,               # connecting rod
        m3, m3                    # piston
    ])

    W_case = np.linalg.inv(M_case)

    def engine_case(t, state):
        """
        Governing equations for this flywheel inertia case.
        """
        q, dq = np.split(state, 2)

        # Evaluate constraints and Jacobian terms
        J_num = np.asarray(J_fn(q, dq), dtype=float)
        dJ_num = np.asarray(dJ_fn(q, dq), dtype=float)

        C_num = np.asarray(C_fn(q, dq), dtype=float).reshape(-1, 1)
        dC_num = np.asarray(dC_fn(q, dq), dtype=float).reshape(-1, 1)

        dq_col = dq.reshape(-1, 1)

        # Gravity and crankshaft resistance
        Q_num = np.asarray(Q_base_fn(q, dq), dtype=float).reshape(-1, 1)

        # Add Otto-cycle gas force to piston vertical coordinate y3
        F_gas = otto_piston_force(q[2], q[7])
        Q_num[7, 0] += F_gas

        # Constraint-force matrix using the current flywheel inertia
        JWJT_case = J_num @ W_case @ J_num.T

        RHS_num = (
            -dJ_num @ dq_col
            -J_num @ W_case @ Q_num
            -(baumgarte_frequency**2) * C_num
            -2 * baumgarte_damping_ratio * baumgarte_frequency * dC_num
        )

        # Solve for constraint reactions
        lam = np.linalg.solve(JWJT_case, RHS_num)

        Q_constraint = J_num.T @ lam

        # Generalised accelerations
        ddq = W_case @ (Q_num + Q_constraint)

        return np.concatenate((dq, ddq.flatten()))

    # Solve this flywheel case
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
        raise RuntimeError(
            f"Solver failed for flywheel inertia I_f = {I_f_case:.5f} kg m^2"
        )

    # Crankshaft angular velocity
    omega_case = sol_case.y[10]

    # Crankshaft angular acceleration.
    # Index 10 of the derivative corresponds to ddot(theta_1).
    alpha_case = np.array([
        engine_case(sol_case.t[i], sol_case.y[:, i])[10]
        for i in range(len(sol_case.t))
    ])

    # Position constraint error
    constraint_case = np.array([
        np.max(
            np.abs(
                np.asarray(
                    C_fn(sol_case.y[:8, i], sol_case.y[8:, i]),
                    dtype=float
                )
            )
        )
        for i in range(len(sol_case.t))
    ])

    return sol_case, omega_case, alpha_case, constraint_case


# -----------------------------
# 20.1 Run all flywheel cases
# -----------------------------

case_results = []
case_data = {}

for I_f_case in I_flywheel_cases:

    sol_case, omega_case, alpha_case, constraint_case = \
        solve_flywheel_case(I_f_case)

    # Equivalent solid-disk flywheel mass at current radius
    # I_f = 0.5*m_f*R_f^2  =>  m_f = 2*I_f/R_f^2
    m_f_case = 2 * I_f_case / R_flywheel**2

    # Angular speed metrics
    omega_max = np.max(omega_case)
    omega_min = np.min(omega_case)
    omega_mean = np.mean(omega_case)
    delta_omega = omega_max - omega_min

    # Coefficient of speed fluctuation
    Cs = delta_omega / abs(omega_mean)

    # Angular acceleration metrics
    alpha_max = np.max(alpha_case)
    alpha_min = np.min(alpha_case)
    alpha_abs_max = np.max(np.abs(alpha_case))

    # Numerical validity metric
    max_constraint_error = np.max(constraint_case)

    case_results.append([
        I_f_case,
        m_f_case,
        omega_max,
        omega_min,
        omega_mean,
        delta_omega,
        Cs,
        alpha_max,
        alpha_min,
        alpha_abs_max,
        max_constraint_error
    ])

    case_data[I_f_case] = {
        'sol': sol_case,
        'omega': omega_case,
        'alpha': alpha_case,
        'constraint_error': constraint_case
    }


case_results = np.array(case_results)

# Column meanings:
# 0  = I_f
# 1  = mass
# 2  = omega_max
# 3  = omega_min
# 4  = omega_mean
# 5  = delta_omega
# 6  = Cs
# 7  = alpha_max
# 8  = alpha_min
# 9  = max absolute alpha
# 10 = max constraint error


# -----------------------------
# 20.2 Percentage reductions
# -----------------------------

delta_omega_no_flywheel = case_results[0, 5]
Cs_no_flywheel = case_results[0, 6]
alpha_no_flywheel = case_results[0, 9]

delta_omega_reduction = (
    (delta_omega_no_flywheel - case_results[:, 5])
    / delta_omega_no_flywheel
) * 100

Cs_reduction = (
    (Cs_no_flywheel - case_results[:, 6])
    / Cs_no_flywheel
) * 100

alpha_reduction = (
    (alpha_no_flywheel - case_results[:, 9])
    / alpha_no_flywheel
) * 100


# -----------------------------
# 20.3 Print results table
# -----------------------------

print('\n' + '=' * 140)
print('FLYWHEEL PARAMETRIC STUDY RESULTS')
print('=' * 140)

print(
    f"{'I_f [kg m^2]':>13}"
    f"{'Mass [kg]':>12}"
    f"{'w_max':>11}"
    f"{'w_min':>11}"
    f"{'w_mean':>11}"
    f"{'Delta w':>12}"
    f"{'C_s':>11}"
    f"{'C_s red [%]':>14}"
    f"{'|alpha|max':>14}"
    f"{'alpha red [%]':>15}"
)

print('-' * 140)

for i in range(len(I_flywheel_cases)):
    print(
        f"{case_results[i, 0]:13.5f}"
        f"{case_results[i, 1]:12.3f}"
        f"{case_results[i, 2]:11.4f}"
        f"{case_results[i, 3]:11.4f}"
        f"{case_results[i, 4]:11.4f}"
        f"{case_results[i, 5]:12.4f}"
        f"{case_results[i, 6]:11.4f}"
        f"{Cs_reduction[i]:14.2f}"
        f"{case_results[i, 9]:14.4f}"
        f"{alpha_reduction[i]:15.2f}"
    )

print('=' * 140)

# Warn if any system stalls or reverses
for i, I_f_case in enumerate(I_flywheel_cases):
    if case_results[i, 3] <= 0:
        print(
            f"Warning: I_f = {I_f_case:.5f} kg m^2 reaches "
            f"omega_min = {case_results[i, 3]:.4f} rad/s, "
            f"indicating stalling or reversal."
        )


# -----------------------------
# 20.4 Save table as CSV
# -----------------------------

results_to_save = np.column_stack((
    case_results,
    delta_omega_reduction,
    Cs_reduction,
    alpha_reduction
))

csv_header = (
    'I_flywheel_kgm2,'
    'm_flywheel_kg,'
    'omega_max_rads,'
    'omega_min_rads,'
    'omega_mean_rads,'
    'delta_omega_rads,'
    'Cs,'
    'alpha_max_rads2,'
    'alpha_min_rads2,'
    'alpha_abs_max_rads2,'
    'max_constraint_error,'
    'delta_omega_reduction_percent,'
    'Cs_reduction_percent,'
    'alpha_reduction_percent'
)

np.savetxt(
    'flywheel_parametric_results.csv',
    results_to_save,
    delimiter=',',
    header=csv_header,
    comments='',
    fmt='%.8e'
)


# -----------------------------
# 20.5 Angular velocity comparison
# -----------------------------

plt.figure(figsize=(10, 5))

for I_f_case in I_flywheel_cases:
    plt.plot(
        case_data[I_f_case]['sol'].t,
        case_data[I_f_case]['omega'],
        label=rf'$I_f={I_f_case:.3f}$ kg m$^2$'
    )

plt.xlabel('Time [s]')
plt.ylabel(r'Crank angular velocity $\dot{\theta}_1$ [rad/s]')
plt.title('Effect of Flywheel Inertia on Crankshaft Angular Speed')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    'flywheel_angular_velocity_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# -----------------------------
# 20.6 Zoomed angular velocity comparison
# -----------------------------

plt.figure(figsize=(10, 5))

for I_f_case in I_flywheel_cases:
    plt.plot(
        case_data[I_f_case]['sol'].t,
        case_data[I_f_case]['omega'],
        label=rf'$I_f={I_f_case:.3f}$ kg m$^2$'
    )

plt.xlim(0, comparison_t_span[1])
plt.xlabel('Time [s]')
plt.ylabel(r'Crank angular velocity $\dot{\theta}_1$ [rad/s]')
plt.title('Crankshaft Angular Speed Comparison')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    'flywheel_angular_velocity_comparison_zoomed.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# -----------------------------
# 20.7 Angular acceleration comparison
# -----------------------------

plt.figure(figsize=(10, 5))

for I_f_case in I_flywheel_cases:
    plt.plot(
        case_data[I_f_case]['sol'].t,
        case_data[I_f_case]['alpha'],
        label=rf'$I_f={I_f_case:.3f}$ kg m$^2$'
    )

plt.xlabel('Time [s]')
plt.ylabel(r'Crank angular acceleration $\ddot{\theta}_1$ [rad/s$^2$]')
plt.title('Effect of Flywheel Inertia on Angular Acceleration')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    'flywheel_angular_acceleration_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# -----------------------------
# 20.8 Coefficient of speed fluctuation graph
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    case_results[:, 0],
    case_results[:, 6],
    marker='o'
)

plt.xlabel(r'Flywheel inertia $I_f$ [kg m$^2$]')
plt.ylabel(r'Coefficient of speed fluctuation $C_s$')
plt.title('Coefficient of Speed Fluctuation Against Flywheel Inertia')
plt.grid()
plt.tight_layout()

plt.savefig(
    'coefficient_of_speed_fluctuation_vs_inertia.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# -----------------------------
# 20.9 Peak angular acceleration graph
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    case_results[:, 0],
    case_results[:, 9],
    marker='o'
)

plt.xlabel(r'Flywheel inertia $I_f$ [kg m$^2$]')
plt.ylabel(r'Maximum $|\ddot{\theta}_1|$ [rad/s$^2$]')
plt.title('Peak Angular Acceleration Against Flywheel Inertia')
plt.grid()
plt.tight_layout()

plt.savefig(
    'peak_angular_acceleration_vs_inertia.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# -----------------------------
# 20.10 Constraint error comparison
# -----------------------------

plt.figure(figsize=(10, 5))

for I_f_case in I_flywheel_cases:
    plt.semilogy(
        case_data[I_f_case]['sol'].t,
        np.maximum(case_data[I_f_case]['constraint_error'], 1e-16),
        label=rf'$I_f={I_f_case:.3f}$ kg m$^2$'
    )

plt.xlabel('Time [s]')
plt.ylabel('Maximum absolute constraint error')
plt.title('Numerical Constraint Error for Flywheel Cases')
plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig(
    'flywheel_constraint_error_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()
