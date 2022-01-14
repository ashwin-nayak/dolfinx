// Copyright (C) 2019 Jorgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "cells.h"
#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <numeric>
#include <stdexcept>
#include <xtensor/xview.hpp>

using namespace dolfinx;
namespace
{
int cell_degree(mesh::CellType type, int num_nodes)
{
  switch (type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return num_nodes - 1;
  case mesh::CellType::triangle:
    switch (num_nodes)
    {
    case 3:
      return 1;
    case 6:
      return 2;
    case 10:
      return 3;
    case 15:
      return 4;
    case 21:
      return 5;
    case 28:
      return 6;
    case 36:
      return 7;
    case 45:
      LOG(WARNING) << "8th order mesh is untested";
      return 8;
    case 55:
      LOG(WARNING) << "9th order mesh is untested";
      return 9;
    default:
      throw std::runtime_error("Unknown triangle layout. Number of nodes: "
                               + std::to_string(num_nodes));
    }
  case mesh::CellType::tetrahedron:
    switch (num_nodes)
    {
    case 4:
      return 1;
    case 10:
      return 2;
    case 20:
      return 3;
    default:
      throw std::runtime_error("Unknown tetrahedron layout.");
    }
  case mesh::CellType::quadrilateral:
  {
    const int n = std::sqrt(num_nodes);
    if (num_nodes != n * n)
    {
      throw std::runtime_error("Quadrilateral of order "
                               + std::to_string(num_nodes) + " not supported");
    }
    return n - 1;
  }
  case mesh::CellType::hexahedron:
    switch (num_nodes)
    {
    case 8:
      return 1;
    case 27:
      return 2;
    default:
      throw std::runtime_error("Unsupported hexahedron layout");
      return 1;
    }
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_triangle(int num_nodes)
{
  switch (num_nodes)
  {
  case 3:
    return {0, 1, 2};
  case 6:
    return {0, 1, 2, 5, 3, 4};
  case 10:
    return {0, 1, 2, 7, 8, 3, 4, 6, 5, 9};
  default:
    throw std::runtime_error("Higher order GMSH triangle not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_tetrahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 4:
    return {0, 1, 2, 3};
  case 10:
    return {0, 1, 2, 3, 9, 6, 8, 7, 4, 5};
  case 20:
    return {0,  1,  2, 3, 14, 15, 8,  9,  13, 12,
            11, 10, 5, 4, 7,  6,  19, 18, 17, 16};
  default:
    throw std::runtime_error("Higher order GMSH tetrahedron not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_hexahedron(int num_nodes)
{
  switch (num_nodes)
  {
  case 8:
    return {0, 1, 3, 2, 4, 5, 7, 6};
  case 27:
    return {0,  1,  3,  2,  4,  5,  7,  6,  8,  9,  10, 11, 12, 13,
            15, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
  default:
    throw std::runtime_error("Higher order GMSH hexahedron not supported");
  }
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> gmsh_quadrilateral(int num_nodes)
{
  switch (num_nodes)
  {
  case 4:
    return {0, 1, 3, 2};
  case 9:
    return {0, 1, 3, 2, 4, 6, 7, 5, 8};
  case 16:
    return {0, 1, 3, 2, 4, 5, 8, 9, 11, 10, 7, 6, 12, 13, 15, 14};
  default:
    throw std::runtime_error("Higher order GMSH quadrilateral not supported");
  }
}
} // namespace
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::perm_vtk(mesh::CellType type,
                                              int num_nodes)
{
  const int degree = cell_degree(type, num_nodes);
  const basix::cell::type basix_type = mesh::cell_type_to_basix_type(type);

  basix::FiniteElement element
      = basix::create_element(basix::element::family::P, basix_type, degree,
                              basix::element::lagrange_variant::equispaced);
  basix::FiniteElement vtk_element
      = basix::create_element(basix::element::family::P, basix_type, degree,
                              basix::element::lagrange_variant::vtk, true);

  xt::xtensor<double, 2> i_mat
      = compute_interpolation_operator(element, vtk_element);

  // Check that matrix is a permutation
  for (std::size_t i = 0; i < i_mat.shape(0); ++i)
  {
    int count = 0;
    for (std::size_t j = 0; j < i_mat.shape(1); ++j)
    {
      if (xt::isclose(i_mat(i, j), 1)())
        count += 1;
      else if (!xt::isclose(i_mat(i, j), 0)())
        throw std::runtime_error("Matrix is not a permutation");
    }
    if (count != 1)
      throw std::runtime_error("Matrix is not a permutation");
  }

  // Compute the permutation
  std::vector<std::uint8_t> perm(i_mat.shape(0));

  for (std::size_t i = 0; i < i_mat.shape(1); ++i)
  {
    for (std::size_t j = 0; j < i_mat.shape(0); ++j)
    {
      if (xt::isclose(i_mat(j, i), 1)())
      {
        perm[i] = j;
        break;
      }
    }
  }

  return perm;
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t> io::cells::perm_gmsh(const mesh::CellType type,
                                               const int num_nodes)
{
  std::vector<std::uint8_t> map;
  switch (type)
  {
  case mesh::CellType::point:
    map = {0};
    break;
  case mesh::CellType::interval:
    map.resize(num_nodes);
    std::iota(map.begin(), map.end(), 0);
    break;
  case mesh::CellType::triangle:
    map = gmsh_triangle(num_nodes);
    break;
  case mesh::CellType::tetrahedron:
    map = gmsh_tetrahedron(num_nodes);
    break;
  case mesh::CellType::quadrilateral:
    map = gmsh_quadrilateral(num_nodes);
    break;
  case mesh::CellType::hexahedron:
    map = gmsh_hexahedron(num_nodes);
    break;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return io::cells::transpose(map);
}
//-----------------------------------------------------------------------------
std::vector<std::uint8_t>
io::cells::transpose(const std::vector<std::uint8_t>& map)
{
  std::vector<std::uint8_t> transpose(map.size());
  for (std::size_t i = 0; i < map.size(); ++i)
    transpose[map[i]] = i;
  return transpose;
}
//-----------------------------------------------------------------------------
xt::xtensor<std::int64_t, 2>
io::cells::compute_permutation(const xt::xtensor<std::int64_t, 2>& cells,
                               const std::vector<std::uint8_t>& p)
{
  xt::xtensor<std::int64_t, 2> cells_new(cells.shape());
  for (std::size_t c = 0; c < cells_new.shape(0); ++c)
  {
    auto cell = xt::row(cells, c);
    auto cell_new = xt::row(cells_new, c);
    for (std::size_t i = 0; i < cell_new.shape(0); ++i)
      cell_new[i] = cell[p[i]];
  }
  return cells_new;
}
//-----------------------------------------------------------------------------
std::int8_t io::cells::get_vtk_cell_type(const dolfinx::mesh::Mesh& mesh,
                                         int dim)
{
  if (mesh.topology().cell_type() == mesh::CellType::prism)
    throw std::runtime_error("More work needed for prism cell");

  // Get cell type
  mesh::CellType cell_type
      = mesh::cell_entity_type(mesh.topology().cell_type(), dim, 0);

  // Determine VTK cell type (Using arbitrary Lagrange elements)
  // https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
  switch (cell_type)
  {
  case mesh::CellType::point:
    return 1;
  case mesh::CellType::interval:
    return 68;
  case mesh::CellType::triangle:
    return 69;
  case mesh::CellType::quadrilateral:
    return 70;
  case mesh::CellType::tetrahedron:
    return 71;
  case mesh::CellType::hexahedron:
    return 72;
  default:
    throw std::runtime_error("Unknown cell type");
  }
}
//----------------------------------------------------------------------------
