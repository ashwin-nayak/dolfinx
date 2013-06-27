// Copyright (C) 2012 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-02-03
// Last changed: 2012-02-16

#ifdef HAS_CGAL

#include <vector>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_with_info_3.h>
#include <CGAL/make_mesh_3.h>

#include <CGAL/Implicit_surface_3.h>
#include <CGAL/Implicit_mesh_domain_3.h>
#include <CGAL/Mesh_domain_with_polyline_features_3.h>

#include <dolfin/common/MPI.h>
#include <dolfin/geometry/ImplicitSurface.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include "CGALMeshBuilder.h"

#include "ImplicitDomainMeshGenerator.h"

// CGAL kernel typedefs
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;

// Point
typedef K::Point_3 Point_3;

typedef CGAL::Robust_weighted_circumcenter_filtered_traits_3<K>
Geom_traits_test;

// Call-back function
typedef boost::function<double (Point_3)> Function;

// Mesh domain
typedef CGAL::Implicit_mesh_domain_3<const Function, K> Mesh_domain_test0;
typedef CGAL::Mesh_domain_with_polyline_features_3<Mesh_domain_test0>
Mesh_domain_test;

// CGAL 3D triangulation vertex typedefs
typedef CGAL::Triangulation_vertex_base_3<Geom_traits_test> Tvb3test_base_test;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, Geom_traits_test,
                                             Tvb3test_base_test> Tvb3test_test;
typedef CGAL::Mesh_vertex_base_3<Geom_traits_test, Mesh_domain_test,
                                 Tvb3test_test> Vertex_base_test;

// CGAL 3D triangulation cell typedefs
typedef CGAL::Triangulation_cell_base_3<Geom_traits_test> Tcb3test_base_test;
typedef CGAL::Triangulation_cell_base_with_info_3<int, Geom_traits_test,
                                            Tcb3test_base_test> Tcb3test_test;
typedef CGAL::Mesh_cell_base_3<Geom_traits_test, Mesh_domain_test,
                               Tcb3test_test> Cell_base_test;

// CGAL 3D triangulation typedefs
typedef CGAL::Triangulation_data_structure_3<Vertex_base_test, Cell_base_test>
Tds_mesh_test;
typedef CGAL::Regular_triangulation_3<Geom_traits_test, Tds_mesh_test> Tr_test;

// Triangulation
//typedef CGAL::Mesh_triangulation_3<Mesh_domain_test>::type Tr_test;

//typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr_test> C3t3_test;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr_test> C3t3_test;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr_test> Mesh_criteria_test;









using namespace dolfin;

// Lightweight class for wrapping callback to ImplicitSurface::operator()
class ImplicitSurfaceWrapper
{
public:
  ImplicitSurfaceWrapper(const ImplicitSurface& surface) : _surface(surface) {}
  ~ImplicitSurfaceWrapper() {}

  double f(const Point_3& p)
  {
    _p[0] = p[0]; _p[1] = p[1]; _p[2] = p[2];
    return _surface(_p);
  }

private:
  Point _p;
  const ImplicitSurface& _surface;
};

//-----------------------------------------------------------------------------
void ImplicitDomainMeshGenerator::generate(Mesh& mesh,
                                           const ImplicitSurface& surface,
                                           double cell_size)
{
  /*
  if (MPI::process_number() == 0)
  {
    ImplicitSurfaceWrapper surface_wrapper(surface);
    boost::function<double (Point_3)> f(boost::bind(&ImplicitSurfaceWrapper::f,
                                                    &surface_wrapper, _1));


    // Create CGAL bounding sphere
    const Point c = surface.sphere.c;
    dolfin_assert(surface.sphere.r > 0.0);
    K::Sphere_3 bounding_sphere(Point_3(c[0], c[1], c[2]),
                                surface.sphere.r*surface.sphere.r);

    // Domain (Warning: Sphere_3 constructor uses squared radius !)
    Mesh_domain_test domain(f, bounding_sphere, 1.0e-5);

    typedef std::vector<Point_3> Polyline_3;
    typedef std::list<Polyline_3> Polylines;

    Polyline_3 polyline;
    polyline.push_back(Point_3(-0.5, 0.0,  0.5));
    polyline.push_back(Point_3( 0.5, 0.0,  0.5));
    polyline.push_back(Point_3( 0.5, 0.0, -0.5));
    polyline.push_back(Point_3(-0.5, 0.0, -0.5));
    polyline.push_back(polyline.front());

    Polylines polylines;
    polylines.push_back(polyline);

    domain.add_features(polylines.begin(), polylines.end());

    // Mesh criteria
    Mesh_criteria_test criteria(CGAL::parameters::facet_angle=30,
                                CGAL::parameters::facet_size=0.5*cell_size,
                                CGAL::parameters::facet_distance=0.1*cell_size,
                                CGAL::parameters::cell_size=cell_size);

    // Mesh generation
    C3t3_test c3t3 = CGAL::make_mesh_3<C3t3_test>(domain, criteria);

    // Build DOLFIN mesh from CGAL 3D mesh/triangulation
    CGALMeshBuilder::build_from_mesh(mesh, c3t3);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
  */
}
//-----------------------------------------------------------------------------
void ImplicitDomainMeshGenerator::generate_surface(Mesh& mesh,
                                                const ImplicitSurface& surface,
                                                double cell_size)

{
  if (MPI::process_number() == 0)
  {
    ImplicitSurfaceWrapper surface_wrapper(surface);
    boost::function<double (Point_3)> f(boost::bind(&ImplicitSurfaceWrapper::f,
                                                    &surface_wrapper, _1));

    // Create CGAL bounding sphere
    const Point c = surface.sphere.c;
    dolfin_assert(surface.sphere.r > 0.0);
    K::Sphere_3 bounding_sphere(Point_3(c[0], c[1], c[2]),
                                surface.sphere.r*surface.sphere.r);

    Mesh_domain_test domain(f, bounding_sphere, 1.0e-5);

    typedef std::vector<Point_3> Polyline_3;
    typedef std::list<Polyline_3> Polylines;

    // Create edge that we want to preserve
    Polylines polylines(1);
    Polyline_3& polyline = polylines.front();
    for(int i = 0; i < 360; ++i)
    {
      Point_3 p(1, std::cos(i*CGAL_PI/180), std::sin(i*CGAL_PI/180));
      polyline.push_back(p);
    }
    polyline.push_back(polyline.front()); // close the line
    domain.add_features(polylines.begin(), polylines.end());

    /*
    Polyline_3 polyline;
    polyline.push_back(Point_3(-0.5, 0.0,  0.5));
    polyline.push_back(Point_3( 0.5, 0.0,  0.5));
    polyline.push_back(Point_3( 0.5, 0.0, -0.5));
    polyline.push_back(Point_3(-0.5, 0.0, -0.5));
    polyline.push_back(polyline.front());

    Polylines polylines;
    polylines.push_back(polyline);
    domain.add_features(polylines.begin(), polylines.end());
    */

    // Mesh criteria
    /*
    Mesh_criteria_test criteria(CGAL::parameters::facet_angle=30,
                                CGAL::parameters::facet_size=cell_size,
                                CGAL::parameters::facet_distance=0.1*cell_size,
                                CGAL::parameters::cell_size=0.0);
    */

    Mesh_criteria_test criteria(CGAL::parameters::edge_size = 0.15,
                                CGAL::parameters::facet_angle = 25,
                                CGAL::parameters::facet_size = cell_size,
                                //CGAL::parameters::cell_radius_edge_ratio = 2,
                                CGAL::parameters::cell_size = 0.0);

    /*
    Mesh_criteria_test criteria(CGAL::parameters::edge_size = 0.15,
                                CGAL::parameters::facet_angle = 25,
                                CGAL::parameters::facet_size = 0.15,
                                CGAL::parameters::cell_radius_edge_ratio = 2,
                                CGAL::parameters::cell_size = 0.15);
    */
    // Mesh generation
    C3t3_test c3t3 = CGAL::make_mesh_3<C3t3_test>(domain, criteria);

    // Build surface DOLFIN mesh from CGAL 3D mesh/triangulation
    CGALMeshBuilder::build_surface_mesh_c3t3(mesh, c3t3, &surface);
  }

  // Build distributed mesh
  MeshPartitioning::build_distributed_mesh(mesh);
}
//-----------------------------------------------------------------------------

#endif
