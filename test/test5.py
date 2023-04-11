import dolfinx
import ufl
import numpy as np

# Define the mesh
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 32, 32)

# Define the function space
V = dolfinx.FunctionSpace(mesh, ("CG", 1))

# Define the test and trial functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define the right-hand side function
p = dolfinx.Function(V)
p.interpolate(dolfinx.Constant(mesh, 1.0))

# Define the bilinear form
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Define the linear form
L = -4 * np.pi * p * v * ufl.dx

# Define the boundary conditions (if any)
bc = dolfinx.DirichletBC(V, dolfinx.Constant(mesh, 0.0), "on_boundary")

# Assemble the system and apply the boundary conditions
A = dolfinx.fem.assemble_matrix(a)
b = dolfinx.fem.assemble_vector(L)
bc.apply(A, b)

# Solve the system
phi = dolfinx.Function(V)
solver = dolfinx.LUSolver(MPI.COMM_WORLD)
solver.solve(A, phi.vector(), b)

# Print the solution norm
print("||phi||_L2 = ", dolfinx.fem.norm(phi, "L2"))
