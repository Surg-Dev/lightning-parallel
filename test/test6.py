from matplotlib import pyplot as plt
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

nx, ny = 100, 100
msh = mesh.create_unit_square(
    MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
V = fem.FunctionSpace(msh, ("Lagrange", 1))

boundary_coords = np.array([[0.50, 0.50], [0.25, 0.25]])

bcs = []

msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)


def check_close(x):
    xs = x[0]
    ys = x[1]

    result = np.zeros_like(xs)

    for coord in boundary_coords:
        x_good = np.isclose(xs, coord[0], atol=1/nx)
        y_good = np.isclose(ys, coord[1], atol=1/ny)
        result = np.logical_or(result, np.logical_and(x_good, y_good))

    return result


lightning_pixels = mesh.locate_entities(msh, dim=(msh.topology.dim - 1),
                                        marker=check_close)


dofs = fem.locate_dofs_topological(
    V=V, entity_dim=1, entities=lightning_pixels)
bcs.append(fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V))

facets = mesh.exterior_facet_indices(msh.topology)
dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bcs.append(fem.dirichletbc(value=ScalarType(1), dofs=dofs, V=V))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(V, 0.)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# print(uh.eval([0.9, 0.9, 0], 0))


# print(uh.x.array)
# result = uh.x.array.reshape((nx+1, ny+1))

# result = uh.vector.array.reshape((nx+1, ny+1))
# plt.imshow(result)
# plt.colorbar()
# plt.show()

# with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_function(uh)

try:
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_poisson.png")
    else:
        plotter.show()

except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
