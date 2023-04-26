#ifndef _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
#define _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_

#include "vertex.h"
#include "array"

namespace SLAM_SOLVER {

/* Class Edge Basic declaration. */
template <typename Scalar>
class EdgeBasic {

public:
    EdgeBasic() = default;
    virtual ~EdgeBasic() = default;

    // Edge index.
    virtual const uint32_t GetId() const = 0;

    // Combine vertice for this edge.
    virtual VertexBasic<Scalar> *GetVertex(uint32_t index) = 0;
    virtual void SetVertex(uint32_t index, VertexBasic<Scalar> *vertex_ptr) = 0;

    static uint32_t &global_id() { return global_id_; }

private:
    // Global index for every edge.
    static uint32_t global_id_;

};

/* Class Edge Basic Definition. */
template<typename Scalar> uint32_t EdgeBasic<Scalar>::global_id_ = 0;

/* Class Edge declaration. */
template <typename Scalar, int32_t VerticeNum, int32_t ResidualDim, int32_t ... VerticeSolveDim>
class Edge : public EdgeBasic<Scalar> {

public:
    Edge();
    virtual ~Edge() = default;

    // Edge index.
    virtual const uint32_t GetId() const { return id_; }

    // Combine vertice for this edge.
    virtual VertexBasic<Scalar> *GetVertex(uint32_t index) { return vertice_[index]; }
    virtual void SetVertex(uint32_t index, VertexBasic<Scalar> *vertex_ptr) { vertice_[index] = vertex_ptr; }

private:
    uint32_t id_ = 0;
    Eigen::Matrix<Scalar, ResidualDim, 1> residual_ = Eigen::Matrix<Scalar, ResidualDim, 1>::Zero();
    std::array<VertexBasic<Scalar> *, VerticeNum> vertice_ = {};

};

/* Class Edge Definition. */
template <typename Scalar, int32_t VerticeNum, int32_t ResidualDim, int32_t ... VerticeSolveDim>
Edge<Scalar, VerticeNum, ResidualDim, VerticeSolveDim ...>::Edge() : EdgeBasic<Scalar>() {
    id_ = EdgeBasic<Scalar>::global_id();
    ++EdgeBasic<Scalar>::global_id();
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_EDGE_H_
