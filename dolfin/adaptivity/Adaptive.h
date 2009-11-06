// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-11-02
// Last changed: 2009-11-06

#ifndef __ADAPTIVE_H
#define __ADAPTIVE_H

#include <set>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class FunctionSpace;
  class Function;
  class BoundaryCondition;
  template <class T> class MeshFunction;

  /// This is the base class for objects that should be updated
  /// automatically during mesh refinement. The purpose of this class
  /// is to collect as much of this functionality as possible into a
  /// single class to keep adaptive subclasses simple.
  ///
  /// Subclasses must register/deregister themselves as depending
  /// objects in the constructor/destructor.

  class Adaptive
  {
  public:

    //--- Registration of depending objects ---

    /// Register depending function space
    void register_object(FunctionSpace* function_space) const;

    /// Register depending function
    void register_object(Function* function) const;

    /// Register depending boundary condition
    void register_object(BoundaryCondition* boundary_condition) const;

    //--- Deregistration of depending objects ---

    /// Deregister depending function space
    void deregister_object(FunctionSpace* function_space) const;

    /// Deregister depending function
    void deregister_object(Function* function) const;

    /// Deregister depending boundary condition
    void deregister_object(BoundaryCondition* boundary_condition) const;

  private:

    // Mesh needs to call refine_mesh
    friend class Mesh;

    //--- Refinement of depending objects (iteration) ---

    /// Refine all depending function spaces to new mesh
    void refine_function_spaces(const Mesh& new_mesh);

    /// Refine all depending functions to new function space
    void refine_functions(const FunctionSpace& new_function_space);

    /// Refine all depending boundary conditions to new function space
    void refine_boundary_conditions(const FunctionSpace& new_function_space);

    //--- Refinement of depending objects ---

    /// Refine mesh for given boundary markers (uniform if null)
    void refine_mesh(Mesh& new_mesh,
                     MeshFunction<bool>* cell_markers);

    /// Refine function space to new mesh
    void refine_function_space(FunctionSpace& function_space,
                               const Mesh& new_mesh);

    /// Refine function to new function space
    void refine_function(Function& function,
                         const FunctionSpace& new_function_space);

    /// Refine boundary_condition to new function space
    void refine_boundary_condition(BoundaryCondition& boundary_condition,
                                   const FunctionSpace& new_function_space);

    // List of depending function spaces
    mutable std::set<FunctionSpace*> _function_spaces;

    // List of depending functions
    mutable std::set<Function*> _functions;

    // List of depending boundary conditions
    mutable std::set<BoundaryCondition*> _boundary_conditions;

  };

}

#endif
