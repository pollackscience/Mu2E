######################
Mathematical Formalism
######################

The magnetic fields within the Mu2E experiment, like most particle physics experiments, are static
and contain no sources.  Therefore, the fields must obey Laplace's equation:

.. math::

    \nabla^2\Phi = 0,

where :math:`\Phi` is the scalar magnetic field potential, which implies that:

.. math::

    B_z = \frac{\partial}{\partial z}\Phi,\quad B_r = \frac{\partial}{\partial r}\Phi,\quad B_\phi =
    \frac{\partial}{\partial \phi}\Phi

The solutions to Laplace's equation are separable in a host of different coordinate systems.  Given
the nature of our particular problem, cylindrical coordinates seem like the natural choice, and
yield a nice series solution:

.. math::

    \Phi = \sum_{n=0}^\infty \sum_{m=1}^\infty
    \big(\sin(n\phi+\delta_n)\big)\big(C_{nm}I_n(k_{m}r)+D_{nm}K_n(k_{m}r)\big)
    \big(A_{nm}\sin(k_{m}z)+B_{nm}\cos(k_{m}z)\big),

where :math:`I_n` and :math:`K_n` are the modified bessel functions of the first and second kind,
respectively, and :math:`A-D` are free parameters.

The :math:`K_n` terms can immediately be excluded, because they all diverge as
:math:`r\rightarrow0`.  This means that :math:`D_{nm} =0` and :math:`C_{nm}` can be reabsorbed into
the :math:`A,B` parameters.  In order to guarantee that the scalar field satisfies some constant
boundary conditions at large Z distances, we define :math:`k_m` as:

.. math::

    k_m=\frac{m\pi}{L_{eff}},

where :math:`L_{eff}` is an effective length that is determined empirically.

There is also a valid generic solution for Laplace's equation in cylindrical coordinates that is
expressed in terms of regular bessel functions and hyperbolic trig functions.  But given the
oscillatory nature of the field distributions as a function of Z, it seems more natural to use a
solution that invokes normal trig functions.
