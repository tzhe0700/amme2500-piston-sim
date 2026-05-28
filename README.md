# amme2500-piston-sim
piston simulation with 4 stroke otto cycle and flywheel. 
TO INSTALL IN TERMINAL:
git clone https://github.com/tzhe0700/amme2500-piston-sim.git \\
code.

<p align="center">
  <img src="images/smiling.gif" width="450">
</p>

# Four-Stroke Otto Cycle Piston Engine Simulation

A constrained multibody dynamics simulation of a slider-crank piston engine with:

- Four-stroke Otto-cycle gas force input
- Flywheel rotational inertia
- Lagrange multiplier constraint forces
- Baumgarte constraint stabilisation
- Animated crank, connecting rod and piston motion



## Simulation Results

### Otto-Cycle Gas Force

The gas force shows the high-force power stroke followed by exhaust, intake and compression strokes.

<p align="center">
  <img src="images/four_stroke_otto_cycle_gas_force.png" width="750">
</p>

### Crankshaft Speed with Flywheel

The flywheel stores rotational energy and reduces angular-speed variation during the non-power strokes.

<p align="center">
  <img src="images/crankshaft_angular_speed_with_flywheel.png" width="750">
</p>

### Piston Motion

<p align="center">
  <img src="images/piston_vertical_position.png" width="750">
</p>

### Constraint Accuracy

<p align="center">
  <img src="images/constraint_error_during_simulation.png" width="750">
</p>

## How to Run

Install the dependencies:

```bash
pip install numpy sympy scipy matplotlib
