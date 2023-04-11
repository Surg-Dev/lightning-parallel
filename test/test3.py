from dolfin import *
import numpy as np

# Define mesh and function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# Define Poisson problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=2)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Solve the problem using CG solver
u = Function(V)  # initial guess
solver = PETScKrylovSolver('cg', 'amg')
solver.parameters['relative_tolerance'] = 1e-8
solver.parameters['maximum_iterations'] = 1000
solver.parameters['monitor_convergence'] = True

# Assign previous solution to initial guess
u_prev = Function(V)
u_prev.vector()[:] = np.random.rand(V.dim())
assign(u, u_prev)

# Solve the problem
solver.solve(a, L, u)

# Plot the solution
plot(u)
interactive()
