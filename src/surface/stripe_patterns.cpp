#include "geometrycentral/surface/stripe_patterns.h"

#include "geometrycentral/numerical/linear_solvers.h"
#include "geometrycentral/surface/direction_fields.h"

#include <Eigen/Sparse>


using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace geometrycentral {
namespace surface {
namespace {

// Compute the 1-form \omega_{ij} such as defined in eq.7 of [Knoppel et al. 2015]
double computeOmega(IntrinsicGeometryInterface& geometry, const VertexData<Vector2>& directionField,
                    const VertexData<double>& frequencies, const Edge& e, bool* crossesSheets = nullptr) {

  geometry.requireEdgeLengths();
  geometry.requireVertexIndices();
  geometry.requireHalfedgeVectorsInVertex();
  geometry.requireTransportVectorsAlongHalfedge();

  // compute roots of direction field (power representation)
  Vector2 Xi = Vector2::fromAngle(directionField[e.firstVertex()].arg() / 2);
  Vector2 Xj = Vector2::fromAngle(directionField[e.secondVertex()].arg() / 2);

  // check if the directions point the same direction
  Vector2 rij = geometry.transportVectorsAlongHalfedge[e.halfedge()];

  double s = dot(rij * Xi, Xj) > 0 ? 1 : -1;
  *crossesSheets = (s < 0);

  // compute the 1-form value along edge ij
  double lij = geometry.edgeLengths[e];
  double phiI = Xi.arg();
  double phiJ = (s * Xj).arg();

  // get the angle of the edge w.r.t. the endpoints' bases
  double thetaI = geometry.halfedgeVectorsInVertex[e.halfedge()].arg();
  double thetaJ = thetaI + rij.arg();

  double omegaIJ = (lij / 2.) * (frequencies[e.firstVertex()] * cos(phiI - thetaI) +
                                 frequencies[e.secondVertex()] * cos(phiJ - thetaJ));

  return omegaIJ;
}

// Build a Laplace-like matrix with double entries (necessary to represent complex conjugation)
SparseMatrix<double> buildVertexEnergyMatrix(IntrinsicGeometryInterface& geometry,
                                             const VertexData<Vector2>& directionField,
                                             const FaceData<int>& branchIndices,
                                             const VertexData<double>& frequencies) {
  SurfaceMesh& mesh = geometry.mesh;

  geometry.requireVertexIndices();
  geometry.requireHalfedgeCotanWeights();

  std::vector<Eigen::Triplet<double>> triplets;
  for (Edge e : mesh.edges()) {
    // compute the discrete 1-form
    bool crossesSheet;
    double omegaIJ = computeOmega(geometry, directionField, frequencies, e, &crossesSheet);

    // compute the cotan weight
    double w = 0;
    if (branchIndices[e.halfedge().face()] == 0) {
      w += geometry.halfedgeCotanWeights[e.halfedge()];
    }
    if (!e.isBoundary() && branchIndices[e.halfedge().twin().face()] == 0) {
      w += geometry.halfedgeCotanWeights[e.halfedge().twin()];
    }

    int i = 2 * geometry.vertexIndices[e.halfedge().vertex()];
    int j = 2 * geometry.vertexIndices[e.halfedge().twin().vertex()];

    // add the diagonal terms
    triplets.emplace_back(i + 0, i + 0, w);
    triplets.emplace_back(i + 1, i + 1, w);

    triplets.emplace_back(j + 0, j + 0, w);
    triplets.emplace_back(j + 1, j + 1, w);

    // compute the new transport coefficient
    Vector2 rij = w * Vector2::fromAngle(omegaIJ);

    // these terms are the same in both cases
    triplets.emplace_back(i + 0, j + 0, -rij.x);
    triplets.emplace_back(i + 1, j + 0, rij.y);

    triplets.emplace_back(j + 0, i + 0, -rij.x);
    triplets.emplace_back(j + 0, i + 1, rij.y);

    // if both vectors don't point the same direction, the block represents conjugation as well as multiplication
    if (crossesSheet) rij *= -1;

    triplets.emplace_back(i + 0, j + 1, -rij.y);
    triplets.emplace_back(i + 1, j + 1, -rij.x);

    triplets.emplace_back(j + 1, i + 0, -rij.y);
    triplets.emplace_back(j + 1, i + 1, -rij.x);
  }

  // assemble matrix from triplets
  SparseMatrix<double> vertexEnergyMatrix(2 * mesh.nVertices(), 2 * mesh.nVertices());
  vertexEnergyMatrix.setFromTriplets(triplets.begin(), triplets.end());

  // Shift to avoid singularity
  SparseMatrix<double> eye(2 * mesh.nVertices(), 2 * mesh.nVertices());
  eye.setIdentity();
  vertexEnergyMatrix += 1e-4 * eye;

  return vertexEnergyMatrix;
}

// Build a lumped mass matrix with double entries
SparseMatrix<double> computeRealVertexMassMatrix(IntrinsicGeometryInterface& geometry) {

  SurfaceMesh& mesh = geometry.mesh;

  geometry.requireVertexIndices();
  geometry.requireVertexDualAreas();

  std::vector<Eigen::Triplet<double>> triplets;
  for (Vertex v : mesh.vertices()) {
    double area = geometry.vertexDualAreas[v];

    size_t i = geometry.vertexIndices[v];
    triplets.emplace_back(2 * i, 2 * i, area);
    triplets.emplace_back(2 * i + 1, 2 * i + 1, area);
  }

  // assemble matrix from triplets
  SparseMatrix<double> realVertexMassMatrix(2 * mesh.nVertices(), 2 * mesh.nVertices());
  realVertexMassMatrix.setFromTriplets(triplets.begin(), triplets.end());

  return realVertexMassMatrix;
}

// Solve the generalized eigenvalue problem in equation 9 [Knoppel et al. 2015]
VertexData<Vector2> computeParameterization(IntrinsicGeometryInterface& geometry,
                                            const VertexData<Vector2>& directionField,
                                            const FaceData<int>& branchIndices, const VertexData<double>& frequencies) {

  SurfaceMesh& mesh = geometry.mesh;

  geometry.requireVertexIndices();

  // Compute vertex energy matrix A and mass matrix B
  SparseMatrix<double> energyMatrix = buildVertexEnergyMatrix(geometry, directionField, branchIndices, frequencies);
  SparseMatrix<double> massMatrix = computeRealVertexMassMatrix(geometry);

  // Find the smallest eigenvector
  Vector<double> solution = smallestEigenvectorPositiveDefinite(energyMatrix, massMatrix);

  // Copy the result to a VertexData vector
  VertexData<Vector2> toReturn(mesh);
  for (Vertex v : mesh.vertices()) {
    toReturn[v].x = solution(2 * geometry.vertexIndices[v]);
    toReturn[v].y = solution(2 * geometry.vertexIndices[v] + 1);
    toReturn[v] = toReturn[v].normalize();
  }
  return toReturn;
}

// extract the final texture coordinates from the parameterization
std::tuple<CornerData<double>, FaceData<int>> computeTextureCoordinates(IntrinsicGeometryInterface& geometry,
                                                                        const VertexData<Vector2>& directionField,
                                                                        const VertexData<double>& frequencies,
                                                                        const VertexData<Vector2>& parameterization) {

  SurfaceMesh& mesh = geometry.mesh;

  CornerData<double> textureCoordinates(mesh);
  FaceData<int> paramIndices(mesh);

  for (Face f : mesh.faces()) {
    // grab the halfedges
    Halfedge hij = f.halfedge();
    Halfedge hjk = hij.next();
    Halfedge hki = hjk.next();

    // grab the parameter values at vertices
    Vector2 psiI = parameterization[hij.vertex()];
    Vector2 psiJ = parameterization[hjk.vertex()];
    Vector2 psiK = parameterization[hki.vertex()];

    double cIJ = (hij.edge().halfedge() != hij ? -1 : 1);
    double cJK = (hjk.edge().halfedge() != hjk ? -1 : 1);
    double cKI = (hki.edge().halfedge() != hki ? -1 : 1);

    // grab the connection coeffients
    bool crossesSheetsIJ, crossesSheetsJK, crossesSheetsKI;
    double omegaIJ = cIJ * computeOmega(geometry, directionField, frequencies, hij.edge(), &crossesSheetsIJ);
    double omegaJK = cJK * computeOmega(geometry, directionField, frequencies, hjk.edge(), &crossesSheetsJK);
    double omegaKI = cKI * computeOmega(geometry, directionField, frequencies, hki.edge(), &crossesSheetsKI);

    if (crossesSheetsIJ) {
      psiJ = psiJ.conj();
      omegaIJ *= cIJ;
      omegaJK *= -cJK;
    }

    if (crossesSheetsKI) {
      psiK = psiK.conj();
      omegaKI *= -cKI;
      omegaJK *= cJK;
    }

    // construct complex transport coefficients
    Vector2 rij = Vector2::fromAngle(omegaIJ);
    Vector2 rjk = Vector2::fromAngle(omegaJK);
    Vector2 rki = Vector2::fromAngle(omegaKI);

    // compute the angles at the triangle corners closest to the target omegas
    double alphaI = psiI.arg();
    double alphaJ = alphaI + omegaIJ - (rij * psiI / psiJ).arg();
    double alphaK = alphaJ + omegaJK - (rjk * psiJ / psiK).arg();
    double alphaL = alphaK + omegaKI - (rki * psiK / psiI).arg();

    // store the coordinates
    textureCoordinates[hij.corner()] = alphaI;
    textureCoordinates[hjk.corner()] = alphaJ;
    textureCoordinates[hki.corner()] = alphaK;
    paramIndices[f] = std::round((alphaL - alphaI) / (2 * PI));
  }
  return std::tie(textureCoordinates, paramIndices);
}

} // namespace


std::tuple<CornerData<double>, FaceData<int>, FaceData<int>>
computeStripePattern(IntrinsicGeometryInterface& geometry, const VertexData<double>& frequencies,
                     const VertexData<Vector2>& directionField) {
  // find singularities of the direction field
  FaceData<int> branchIndices = computeFaceIndex(geometry, directionField, 2);

  // solve the eigenvalue problem (multiply by 2pi to get the right frequencies)
  VertexData<Vector2> parameterization =
      computeParameterization(geometry, directionField, branchIndices, 2 * PI * frequencies);

  // compute the final corner-based values, along with singularities of the stripe pattern
  CornerData<double> textureCoordinates;
  FaceData<int> zeroIndices;
  std::tie(textureCoordinates, zeroIndices) =
      computeTextureCoordinates(geometry, directionField, 2 * PI * frequencies, parameterization);

  return std::tie(textureCoordinates, zeroIndices, branchIndices);
}

namespace {

// checks if there's k \in \mathbb{Z} such that val1 < 2 k \pi < val2 or val2 < 2 k \pi < val1
bool crossesModulo2Pi(double val1, double val2, double& bary) {
  if (val1 == val2) return false;

  if (val1 < val2) {
    double isoval = 2 * PI * std::ceil(val1 / (2 * PI));
    if (val2 > isoval) {
      bary = (isoval - val2) / (val1 - val2);
      return true;
    }
  } else {
    double isoval = 2 * PI * std::ceil(val2 / (2 * PI));
    if (val1 > isoval) {
      bary = (isoval - val2) / (val1 - val2);
      return true;
    }
  }
  return false;
}

} // namespace

std::vector<Isoline> extractIsolinesFromStripePattern(IntrinsicGeometryInterface& geometry,
                                                      const CornerData<double>& stripeValues,
                                                      const FaceData<int>& stripesIndices,
                                                      const FaceData<int>& fieldIndices) {
  SurfaceMesh& mesh = geometry.mesh;

  geometry.requireFaceIndices();

  std::vector<Isoline> isolines;
  FaceData<bool> visited(mesh, false);

  for (Face f : mesh.faces()) {
    if (visited[f] || stripesIndices[f] != 0 || fieldIndices[f] != 0) continue;
    visited[f] = true;

    Isoline iso;
    iso.open = true;
    int nbOfPieces = 0;
    for (Halfedge h : f.adjacentHalfedges()) {
      double bary;
      if (crossesModulo2Pi(stripeValues[h.corner()], stripeValues[h.next().corner()], bary)) {
        nbOfPieces += 1;
        std::vector<std::pair<Halfedge, double>> isoPts;
        isoPts.emplace_back(h, bary);

        Face prevFace = f;
        Face curFace = h.twin().face();
        bool done = false;
        while (!curFace.isBoundaryLoop() && !done && stripesIndices[curFace] == 0 && fieldIndices[curFace] == 0) {
          visited[curFace] = true;
          done = true;
          for (Halfedge he : curFace.adjacentHalfedges()) {
            Face oppFace = he.twin().face();
            if (oppFace == prevFace) // don't examine the shared edge twice
              continue;

            if (crossesModulo2Pi(stripeValues[he.corner()], stripeValues[he.next().corner()], bary)) {
              if (!oppFace.isBoundaryLoop() && visited[oppFace]) {
                done = true;
                if (oppFace == f) iso.open = false; // went back to the original face and made a loop
              } else {
                if (oppFace.isBoundaryLoop() || stripesIndices[oppFace] != 0 || fieldIndices[oppFace] != 0)
                  done = true;
                else
                  done = false;

                isoPts.emplace_back(he, bary);
                prevFace = curFace;
                curFace = oppFace;
              }
              break;
            }
          }
        }
        if (iso.barycenters.size() == 0) // reverse the order of the elements
          for (auto it = isoPts.rbegin(); it != isoPts.rend(); ++it) iso.barycenters.push_back(*it);
        else // stitch together both traces into one isoline
          for (auto& it : isoPts) iso.barycenters.push_back(it);
      }
    }
    if (nbOfPieces > 0)
      isolines.push_back(iso);

    if (nbOfPieces > 2) // isolines stop at singularities, so they should never branch out
      throw std::runtime_error("Isolines should only branch out on singularities");
  }
  return isolines;
}

std::tuple<std::vector<Vector3>, std::vector<std::array<int, 2>>>
extractPolylinesFromStripePattern(EmbeddedGeometryInterface& geometry, const CornerData<double>& values,
                                  const FaceData<int>& stripesIndices, const FaceData<int>& fieldIndices) {
  auto isolines = extractIsolinesFromStripePattern(geometry, values, stripesIndices, fieldIndices);

  geometry.requireVertexPositions();
  std::vector<Vector3> points;
  std::vector<std::array<int, 2>> edges;

  int i = 0;
  for (auto isoline : isolines) {
    int start = i;
    for (auto pair : isoline.barycenters) {
      Halfedge h = pair.first;
      double bary = pair.second;
      points.push_back(bary * geometry.vertexPositions[h.tailVertex()] +
                       (1 - bary) * geometry.vertexPositions[h.tipVertex()]);
      // make sure not to add an edge in the last iteration
      if (i < start + static_cast<int>(isoline.barycenters.size()) - 1) {
        std::array<int, 2> edge = {i, i + 1};
        edges.push_back(edge);
      }
      i += 1;
    }
    if (!isoline.open) // close loop
    {
      std::array<int, 2> edge = {i - 1, start};
      edges.push_back(edge);
    }
  }

  return std::tie(points, edges);
}

} // namespace surface
} // namespace geometrycentral