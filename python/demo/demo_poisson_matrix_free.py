# # Matrix-free Poisson solver
#
# Copyright (C) 2022 Adeeb Arif Kor
#
# This demo shows how to solve the Poisson problem using a matrix-free method.
# In particular, we build a matrix free operator and use the PETSc CG solver to
# solve the corresponding linear system of equation.

import numpy as np

from dolfinx.cpp.fem import pack_constants, pack_coefficients
from dolfinx.fem import (Constant, FunctionSpace, Function, assemble_scalar,
                         dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices
from ufl import TrialFunction, TestFunction, action, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc


class MatFree:
    """
    Matrix-free operator to use in the PETSc solver.
    """

    def __init__(self, M, ui, bc):
        self.M = M
        self.ui = ui
        self.consts = pack_constants(form(self.M))
        self.bc = bc

    def mult(self, mat, x, y):
        y.set(0.0)

        # ui <- x
        x.copy(result=self.ui.vector)

        # y <- A x
        coeffs = pack_coefficients(form(self.M))
        assemble_vector(y, form(self.M), self.consts, coeffs)

        # Set BC dofs to zero
        set_bc(y, [self.bc], None, 0.0)

        y.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        y.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# Create mesh and function space
mesh = create_rectangle(comm=MPI.COMM_WORLD,
                        points=((0.0, 0.0), (1.0, 1.0)), n=(10, 10),
                        cell_type=CellType.triangle)
V = FunctionSpace(mesh, ("Lagrange", 2))

# Create the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define the source function
f = Constant(mesh, PETSc.ScalarType(-6.0))

# Define the variational form
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

# Define the action of the bilinear form "a" on a function ui
ui = Function(V)
M = action(a, ui)

# Specify boundary condition
u_D = Function(V)
u_D.interpolate(lambda x: 1 + x[0] * x[0] + 2 * x[1] * x[1])

mesh.topology.create_connectivity(1, 2)
facets = exterior_facet_indices(mesh.topology)
bdofs = locate_dofs_topological(V, 1, facets)
bc = dirichletbc(u_D, bdofs)

# Assemble RHS vector
b = assemble_vector(form(L))
set_bc(ui.vector, [bc], None, -1.0)
assemble_vector(b, form(M))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc], None, 0.0)
b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

# Set up a PETSc Mat object using the matrix free operator
A = PETSc.Mat().create()
Actx = MatFree(M, ui, bc)
A.setSizes(b.getSizes())
A.setType(A.Type.PYTHON)
A.setPythonContext(Actx)
A.setUp()

# Set the linear solver options
opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["pc_type"] = "none"

# Define the PETSc linear solver
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setFromOptions()
solver.setConvergenceHistory()
solver.setOperators(A)

# Solve
uh = Function(V)
solver.solve(b, uh.vector)
set_bc(uh.vector, [bc], None, 1.0)

# Compute error between exact and finite element solution
E = uh - u_D
error = MPI.COMM_WORLD.allreduce(
    assemble_scalar(form(inner(E, E) * dx)), op=MPI.SUM)

print("Number of CG iterations", solver.getIterationNumber())
print("Finite element error (L2 norm (squared))", np.abs(error))
