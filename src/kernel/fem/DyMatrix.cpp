// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Vector.h>
#include <dolfin/Mesh.h>
#include <dolfin/PDE.h>
#include <dolfin/FEM.h>
#include <dolfin/MassMatrix.h>
#include <dolfin/DyMatrix.h>

using namespace dolfin;

namespace dolfin
{

  // The variational form for the derivative
  class DyForm : public PDE
  {
  public:
    
    DyForm() : PDE(3) {}

    real lhs(const ShapeFunction& u, const ShapeFunction& v)
    {
      return ddy(u)*v*dx;
    }

  };
  
}

//-----------------------------------------------------------------------------
DyMatrix::DyMatrix(Mesh& mesh) : Matrix(mesh.noNodes(), mesh.noNodes())
{
  dolfin_error("This function needs to be updated to the new format.");

  /*
  DyForm form;

  // Assemble form for derivative
  FEM::assemble(form, mesh, *this);

  // Compute lumped mass matrix
  MassMatrix M(mesh);
  Vector m;
  M.lump(m);

  // Multiply from left with inverse of lumped mass matrix
  for (unsigned int i = 0; i < size(0); i++)
  {
    for (unsigned int pos = 0; !endrow(i, pos); pos++)
    {
      unsigned int j = 0;
      (*this)(i, j, pos) /= m(i);
    }
  }      
  */
}
//-----------------------------------------------------------------------------
