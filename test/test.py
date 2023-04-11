from fenics import *
import matplotlib.pyplot as plt

# Define the mesh and the function space
nx, ny = 100, 100
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions


def boundary(x, on_boundary):
    return on_boundary


bc = DirichletBC(V, Constant(0), boundary)

# Define the variational problem
phi = TrialFunction(V)
p = Constant(1)
f = Constant(-4*pi*p)
a = inner(grad(phi), grad(TestFunction(V))) * dx
L = f * TestFunction(V) * dx

# Solve the variational problem
phi = Function(V)
solve(a == L, phi, bc)

# Output the solution
plot(phi)
plt.show()
