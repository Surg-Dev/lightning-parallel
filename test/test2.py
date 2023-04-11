import timeit
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

nx, ny = 256, 256


def make_problem():

    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    phi = TrialFunction(V)
    p = Constant(0)
    f = -4 * np.pi * p
    v = TestFunction(V)
    a = inner(grad(phi), grad(v)) * dx
    L = f * v * dx

    return V, a == L


def make_boundaries(V, coords):

    def boundary(x):
        return any(near(x[0], coord[0]/nx) and near(x[1], coord[1]/ny) for coord in coords)

    return [DirichletBC(V, Constant(0), boundary, 'pointwise'),
            DirichletBC(V, Constant(1), 'on_boundary')]


def solve_problem(V, a_L, bcs, phi=None):
    # make this previous answers https://fenicsproject.org/qa/9536/how-to-set-initial-guess-for-nonlinear-variational-problem/
    phi = phi or Function(V)
    print(phi)
    solve(a_L, phi, bcs, solver_parameters={
          'linear_solver': 'cg', 'preconditioner': 'ilu'})
    return phi


V, a_L = make_problem()
coords = [(40, 40)]

start = timeit.default_timer()
bcs = make_boundaries(V, coords)
stop = timeit.default_timer()
print('make_boundaries: ', stop - start)

start = timeit.default_timer()
phi = solve_problem(V, a_L, bcs)
stop = timeit.default_timer()
print('solve_problem: ', stop - start)

plt.imshow(phi.compute_vertex_values().reshape((nx + 1, ny + 1)))
plt.colorbar()
plt.show()
