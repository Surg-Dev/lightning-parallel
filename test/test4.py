from matplotlib import pyplot as plt
from dolfinx import plot
import pyvista
from petsc4py.PETSc import ScalarType
import ufl
from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
import numpy as np

nx, ny = 100, 100
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)

V = fem.FunctionSpace(domain, ("CG", 1))

# uD = fem.Function(V)
# uD.interpolate(lambda x: np.ones_like(x[0]))
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = -4 * np.pi * fem.Constant(domain, ScalarType(0))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# convert uh to ndarray
result = uh.vector.array.reshape((nx+1, ny+1))
plt.imshow(result)
plt.colorbar()
plt.show()
