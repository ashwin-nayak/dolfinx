// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfinx/common/types.h>
#include <memory>
#include <set>
#include <ufc.h>
#include <vector>

namespace dolfinx
{
namespace mesh
{
enum class CellType;
}

namespace fem
{

/// The class represents the degree-of-freedom (dofs) for an element.
/// Dofs are associated with a mesh entity. This class also handles
/// sub-space dofs, which are views into the parent dofs.

// TODO: For this class/concept to be robust, the topology of the
//       reference cell needs to be defined.

// TODO: Handle block dofmaps properly

class ElementDofLayout
{
public:
  /// Constructor
  /// @param[in] block_size The number of dofs co-located at each point.
  /// @param[in] entity_dofs The dofs on each entity, in the format:
  ///   entity_dofs[entity_dim][entity_number] = [dof0, dof1, ...]
  /// @param[in] parent_map TODO
  /// @param[in] sub_dofmaps TODO
  /// @param[in] domain_type The domain of the function space.
  /// @param[in] cell_type The cell type of the mesh.
  ElementDofLayout(
      int block_size,
      const std::vector<std::vector<std::set<int>>>& entity_dofs,
      const std::vector<int>& parent_map,
      const std::vector<std::shared_ptr<const ElementDofLayout>>& sub_dofmaps,
      const mesh::CellType cell_type,
      const mesh::CellType domain_type);

  /// Copy the DOF layout, discarding any parent information
  ElementDofLayout copy() const;

  /// Copy constructor
  ElementDofLayout(const ElementDofLayout& dofmap) = default;

  /// Move constructor
  ElementDofLayout(ElementDofLayout&& dofmap) = default;

  /// Destructor
  ~ElementDofLayout() = default;

  /// Copy assignment
  ElementDofLayout& operator=(const ElementDofLayout& dofmap) = default;

  /// Move assignment
  ElementDofLayout& operator=(ElementDofLayout&& dofmap) = default;

  /// Return the dimension of the local finite element function space on
  /// a cell (number of dofs on element)
  /// @return Dimension of the local finite element function space.
  int num_dofs() const;

  /// Return the number of dofs for a given entity dimension
  /// @param[in] dim Entity dimension
  /// @return Number of dofs associated with given entity dimension
  int num_entity_dofs(int dim) const;

  /// Return the number of closure dofs for a given entity dimension
  /// @param[in] dim Entity dimension
  /// @return Number of dofs associated with closure of given entity
  ///   dimension
  int num_entity_closure_dofs(int dim) const;

  /// Local-local mapping of dofs on entity of cell
  /// @param[in] entity_dim The entity dimension
  /// @param[in] cell_entity_index The local entity index on the cell
  /// @return Degrees of freedom on a single element.
  std::vector<int> entity_dofs(int entity_dim, int cell_entity_index) const;

  /// Local-local closure dofs on entity of cell
  /// @param[in] entity_dim The entity dimension
  /// @param[in] cell_entity_index The local entity index on the cell
  /// @return Degrees of freedom on a single element
  std::vector<int> entity_closure_dofs(int entity_dim,
                                       int cell_entity_index) const;

  /// Direct access to all entity dofs (dof = _entity_dofs[dim][entity][i])
  const std::vector<std::vector<std::set<int>>>& entity_dofs_all() const;

  /// Direct access to all entity closure dofs (dof =
  /// _entity_dofs[dim][entity][i])
  const std::vector<std::vector<std::set<int>>>&
  entity_closure_dofs_all() const;

  /// Get number of sub-dofmaps
  int num_sub_dofmaps() const;

  /// Get sub-dofmap given by list of components, one for each level
  std::shared_ptr<const ElementDofLayout>
  sub_dofmap(const std::vector<int>& component) const;

  /// Get view for a sub dofmap, defined by the component list (as for
  /// sub_dofmap()), into this dofmap. I.e., the dofs in this dofmap
  /// that are the sub-dofs.
  std::vector<int> sub_view(const std::vector<int>& component) const;

  /// Block size
  int block_size() const;

  /// True iff dof map is a view into another map
  ///
  /// @returns bool True if the dof map is a sub-dof map (a view into
  ///   another map).
  bool is_view() const;

  int domain_dim() const;

private:
  // Block size
  int _block_size;

  // Mapping of dofs to this ElementDofLayout's immediate parent
  std::vector<int> _parent_map;

  // Total number of dofs on this element dofmap
  int _num_dofs;

  // The number of dofs associated with each entity type
  std::array<int, 4> _num_entity_dofs;

  // The number of dofs associated with each entity type, including all
  // connected entities of lower dimension.
  std::array<int, 4> _num_entity_closure_dofs;

  // List of dofs per entity, ordered by dimension.
  // dof = _entity_dofs[dim][entity][i]
  std::vector<std::vector<std::set<int>>> _entity_dofs;

  // List of dofs with connected entities of lower dimension
  std::vector<std::vector<std::set<int>>> _entity_closure_dofs;

  // List of sub dofmaps
  std::vector<std::shared_ptr<const ElementDofLayout>> _sub_dofmaps;

  int _domain_dim;
};

} // namespace fem
} // namespace dolfinx
