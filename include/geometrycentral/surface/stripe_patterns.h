#pragma once

#include "geometrycentral/surface/embedded_geometry_interface.h"
#include "geometrycentral/surface/intrinsic_geometry_interface.h"
#include <vector>

namespace geometrycentral {
namespace surface {

/**
 * This function is an implementation of "Stripe Patterns on Surfaces" [Knoppel et al. 2015]
 * It takes as input a geometry along with vertex-based frequencies and a line field (2-RoSy) and outputs a
 * 2\pi-periodic function defined on triangle corners such that the 0 (mod 2\pi) isolines of this function are stripes
 * following the direction field spaced according to the target frequencies
 */
std::tuple<CornerData<double>, FaceData<int>, FaceData<int>>
computeStripePattern(IntrinsicGeometryInterface& geometry, const VertexData<double>& frequencies,
                     const VertexData<Vector2>& directionField);

struct Isoline {
  std::vector<std::pair<Halfedge, double>> barycenters;
  bool open;
};

// extracts isolines as a list of barycentric coordinates and their corresponding halfedges
std::vector<Isoline> extractIsolinesFromStripePattern(IntrinsicGeometryInterface& geometry,
                                                      const CornerData<double>& stripeValues,
                                                      const FaceData<int>& zeroIndices,
                                                      const FaceData<int>& fieldIndices);

// extracts isolines as a list of vertex positions, requires the geometry to have an embedding
std::tuple<std::vector<Vector3>, std::vector<std::array<int, 2>>>
extractPolylinesFromStripePattern(EmbeddedGeometryInterface& geometry, const CornerData<double>& values,
                                  const FaceData<int>& stripesIndices, const FaceData<int>& fieldIndices);

} // namespace surface
} // namespace geometrycentral
