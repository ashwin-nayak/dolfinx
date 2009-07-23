"""This demo illustrates how to use of DOLFIN for solving a nonlinear
PDE, in this case a nonlinear variant of Poisson's equation,

    - div (1 + u^2) grad u(x, y) = f(x, y)

on the unit square with source f given by

     f(x, y) = x*sin(y)

and boundary conditions given by

     u(x, y)     = 1  for x = 0
     du/dn(x, y) = 0  otherwise

This is equivalent to solving

    F(u) = (grad(v), (1 + u^2)*grad(u)) - (v, f) = 0

"""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2008-12-26"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Original implementation (../cpp/main.cpp) by Garth N. Wells.
# Modified by Anders Logg, 2008.
# Modified by Harish Narayanan, 2009.

from dolfin import *

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Create mesh and define function space
mesh = UnitSquare(16, 16)
V = FunctionSpace(mesh, "CG", 1)

# Define boundary condition
g = Constant(mesh, 1)
bc = DirichletBC(V, g, DirichletBoundary())

# Define source and solution functions
f = Function(V, "x[0]*sin(x[1])")
U = Function(V)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
L = inner(grad(v), (1 + U**2)*grad(U))*dx - v*f*dx
a = derivative(L, U, u)

# Solve nonlinear variational problem
problem = VariationalProblem(a, L, bc, nonlinear=True)
problem.solve(U)

# Plot solution and solution gradient
plot(U, title="Solution")
plot(grad(U), title="Solution gradient")
interactive()

# Save solution in VTK format
file = File("nonlinear_poisson.pvd")
file << U
