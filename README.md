# PINNs for inverse estimation for wave speed
### The repository https://github.com/mahindrautela/PINNs-for-wave-propagation and associated paper solves the forward problem where PINNs are used to obtain a surrogate spatial-temporal displacement field in wave propagation problem.

### The inverse problem of estimating wave speed from limited time snapshot measurements is more challenging. One direction to solve the inverse problem is formulating another optimization framework nested over a PINN-based forward solver.

### *main_string_speedscale.py* solve the forward problem and gets a single snapshot measurements at a particular speed. This acts as an experimental obervation.

### *main_string_speedscale_inverse.py* solves the inverse problem of estimating wave speed from noisy experimental observation.
