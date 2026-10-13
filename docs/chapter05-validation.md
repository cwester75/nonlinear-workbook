# Chapter 5 validation

CPython3.12.14, NumPy2.3.5, Matplotlib3.11.2; unchanged frozen dependencies in
requirements-examples.txt. Seed20261013 for observation noise. The full suite has
86 passing tests, preserving all77 previous tests. Reproduction commands are in
chapter05.md. All examples complete; six plots were visually inspected.
Both complete runs produced 19 byte-identical JSON/CSV/PNG artifacts; source PDF,
generated environments and outputs remain excluded from Git.

## Analytical acceptance checks

Approved tolerances precede recording the nonlinear results:

- Centered-difference Jacobians: absolute error<1e-6, step1e-6.
- Lorenz and chemical equilibrium residuals<1e-10.
- Diagonal linear-flow QR and area rates within1e-7; both expanding and contracting
  directions checked. An independent wedge derivative identity checks signs/order.
- Variable-divergence nonlinear fixture x'=-x^2,y'=-y has exact finite-time
  exponents(-log(3),-1) over duration2 from(1,1); trace and rates agree within1e-7.
- Lorenz exponent sum versus -(sigma+1+b): tolerance1e-5.
- Integral derivative residual<1e-12; relative numerical drift<1e-6 on horizon5.
- Lorenz eigenvalue crossing agrees with analytic rH within1e-6.

Tests also cover wrong Jacobian/state shape, singular initial basis, zero initial
area, invalid parameters, overflow, escape, nonfinite RHS, incomplete integration,
ambiguous complex pairs, real crossings and missing brackets. No stochastic
acceptance range is fitted to observed nonlinear exponents.

## Nonlinear QR results

All runs use burn20, identity initial basis, adaptive atol=rtol=1e-9, initial/max
step.02, QR interval.1, escape radius1e6. Rates remain in tangent-column order.

| Model and fixture | Duration100 | Duration200 |
|---|---|---|
| Lorenz sigma16,r40,b4, initial(.8,.8,.8) | (1.3699894,.0068122,-22.3768016) | (1.3801451,-.0037535,-22.3763917) |
| Generalized Lotka a2.9851,b3,c7, initial(1,1.5,2.5) | (-.0596664,-.0816665,-.1367942) | (-.0286431,-.0615282,-.0885459) |
| Four-dimensional a.25,b3,c.5,d.05, initial(-19,0,0,15) | (.1518824,-.0151548,.0092505,-29.8208929) | (.1361277,.0073527,.0037803,-25.5221080) |

Lorenz sum minus average trace is about-2.8e-8; generalized Lotka about4.7e-9;
four-dimensional about-3.5e-6 at100 and-1.8e-6 at200. This supports variational
accounting; it does not establish convergence of individual exponents.

Lorenz duration100 largest rates change to1.3866487 with QR interval.05,
1.3981118 with interval.2, and1.3765551 with interval.1 and atol=rtol=1e-10.
The near-zero flow direction changes sign at finite duration. The book's
approximate1.37 is a sensitivity comparison, not an exact target.

The generalized Lotka trajectory shows strong transient decay over20..70;
all three coordinates remain positive (minima over0..70 about.5347,.6310,1.2266).
The reported finite-duration rates do not support an imposed chaos label.
Four-dimensional rates near zero also remain duration-sensitive; counting every
small positive finite-time value as an asymptotic exponent would be misleading.

## Independent exterior-area comparison

Initial vectors(1,2,3,4) and(4,-1,2,0), burn20, interval.1, same solver controls.

| Duration | Exterior rate | Largest two final QR estimates summed | Difference |
|---|---|---|---|
| 100 | .14828263 | .16113284 | -.01285021 |
| 200 | .14265942 | .14348042 | -.00082100 |

These independent augmented solves can follow different numerical trajectories in
chaotic regimes; no pointwise long-time agreement is presumed. The shrinking gap
is a finite-horizon observation, not a guarantee of convergence.

## Equilibria, integral and data-only estimates

Lorenz sigma16,b4 has analytic rH=368/11=33.45454545; bracketed crossing agrees
within1e-8. Origin stability also changes at r1. Chemical s10,q.01,f1 on the positive
equilibrium branch gives a candidate near w3.0090545021, imaginary eigenfrequency
1.3351230280. Other branches are returned by the API. This is not a proof of a
nondegenerate Hopf bifurcation or a determination of cycle stability.

Time-dependent Lorenz integral: sigma10,r0,b1, initial(1,1,1), horizon5,
atol=rtol=1e-11; observed maximum relative drift1.7612e-11.

32 scalar-observation configurations retain full clean/noisy data, solver work,
Kantz curves/counts and two explicit fit windows. Both systems burn20; sample
counts256/512, intervals.02/.1, noise0/.02, dimensions2/3, delay2, Theiler20,
horizon8, slices[0:3]/[1:6]. Each radius is .1*std(clean,ddof=0), independent of noise.
Rates range1.2873..11.1521 for Lorenz and-.2462..51.9760 for generalized Lotka.
None of these configurations lacks neighbors. These broad ranges demonstrate
finite-window/noise/sampling sensitivity, not agreement with the known-equation
spectrum. Fixed absolute noise can dominate a decaying signal; its amplitude is
reported rather than silently rescaled.
