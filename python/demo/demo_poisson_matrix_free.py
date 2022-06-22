import numpy as np
import matplotlib.pyplot as plt

import ufl
from dolfinx import fem, io, mesh, plot, cpp, common
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                             points=((0.0, 0.0), (2.0, 1.0)), n=(64, 32),
                             cell_type=mesh.CellType.triangle)
V = fem.FunctionSpace(msh, ("Lagrange", 1))

facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                      np.isclose(x[0], 2.0)))

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = fem.dirichletbc(value=PETSc.ScalarType(0), dofs=dofs, V=V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

# b = fem.petsc.assemble_vector(fem.form(L))
# fem.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
# b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# fem.set_bc(b, [bc])

# A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
# A.assemble()

# opts = PETSc.Options()
# opts["ksp_type"] = "cg"
# opts["pc_type"] = "none"

# solver = PETSc.KSP().create(MPI.COMM_WORLD)
# solver.setFromOptions()
# solver.setConvergenceHistory()
# solver.setOperators(A)

# solution = fem.Function(V)
# solver.solve(b, solution.vector)
# solution.x.scatter_forward()
# residuals = solver.getConvergenceHistory()

# with io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5) as file:
#     file.write_mesh(msh)
#     file.write_function(solution)

# plt.semilogy(residuals)
# plt.savefig("residuals.png")

class MatFreeA:
    def __init__(self, a, bc):
        self.a = a
        self.bc = bc
        self.ui = fem.Function(V)
        self.ui.interpolate(lambda x: x[0])
        self.M = fem.form(ufl.action(self.a, self.ui))
        self.consts = cpp.fem.pack_constants(self.M)
        self.coeffs = cpp.fem.pack_coefficients(self.M)

    def mult(self, mat, x, y):
        # ui <- x
        x.copy(result=self.ui.vector)
        self.ui.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                   mode=PETSc.ScatterMode.FORWARD)
        # y <- M x
        fem.petsc.assemble_vector(y, self.M)#, self.consts, self.coeffs)
        y.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        # Set BC dofs to zero
        fem.set_bc(y, [self.bc], 0.0)


b = fem.petsc.assemble_vector(fem.form(L))
fem.


# M = PETSc.Mat().create()
# Mctx = MatFreeA(a, bc)
# M.setSizes(b.getSizes())
# M.setType(M.Type.PYTHON)
# M.setPythonContext(Mctx)
# M.setUp()

# opts = PETSc.Options()
# opts["ksp_type"] = "cg"
# opts["pc_type"] = "none"

# solver = PETSc.KSP().create(MPI.COMM_WORLD)
# solver.setFromOptions()
# solver.setConvergenceHistory()
# solver.setOperators(M)

# solution = fem.Function(V)
# solver.solve(b, solution.vector)
# solution.x.scatter_forward()
# residuals = solver.getConvergenceHistory()

# with io.XDMFFile(MPI.COMM_WORLD, "solution_matfree.xdmf", "w", encoding=io.XDMFFile.Encoding.HDF5) as file:
#     file.write_mesh(msh)
#     file.write_function(solution)

# plt.semilogy(residuals)
# plt.savefig("residual_matfree.png")