import fenics
import timeit
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

nx, ny = 8, 8


mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

phi = TrialFunction(V)
p = Constant(0)
f = -4 * np.pi * p
v = TestFunction(V)
a = inner(grad(phi), grad(v)) * dx
L = f * v * dx

coords = [(2, 2)]


def boundary(x):
    return any(near(x[0], coord[0]/nx) and near(x[1], coord[1]/ny) for coord in coords)


bcs = [DirichletBC(V, Constant(0), boundary, 'pointwise'),
       DirichletBC(V, Constant(1), 'on_boundary')]

phi = Function(V)
# solve(a == L, phi, bcs, solver_parameters={
#     'linear_solver': 'cg', 'preconditioner': 'ilu'})
M = phi * dx
problem = LinearVariationalProblem(a, L, phi, bcs)
solver = AdaptiveLinearVariationalSolver(problem, M)
solver.parameters["error_control"]["dual_variational_solver"]["linear_solver"] = "cg"


# start timer
start = timeit.default_timer()
solver.solve(1e-2)
# stop timer
stop = timeit.default_timer()
print('solve_problem: ', stop - start)

# rasterize it
# plt.imshow(phi.compute_vertex_values().reshape((nx + 1, ny + 1)))

# fenics.plot(phi.leaf_node().function_space().mesh())

# plot(phi)
plt.title("Adapted mesh")

plt.xlabel("$x$")

plt.ylabel("$y$")

plt.show()

# plt.imshow(phi.compute_vertex_values().reshape((nx + 1, ny + 1)))
# plt.colorbar()
# plt.show()
