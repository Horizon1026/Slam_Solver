#ifndef _GENERAL_GRAPH_OPTIMIZOR_VERTEX_H_
#define _GENERAL_GRAPH_OPTIMIZOR_VERTEX_H_

#include "datatype_basic.h"

namespace SLAM_SOLVER {

/* Class Vertex Basic declaration. */
template <typename Scalar>
class VertexBasic {

public:
    VertexBasic() = default;
    virtual ~VertexBasic() = default;

    // Vertex index.
    virtual const uint32_t GetId() = 0;

    // Stored dimension and solve dimension can be different for Quatnion.
    virtual const int32_t GetStoreDimension() const = 0;
    virtual const int32_t GetSolveDimension() const = 0;

    virtual const std::string GetType() = 0;

    // Param operation with scalar adaption.
    virtual void SetParam(const Vec &param) = 0;
    virtual Vec GetParam() = 0;

    // Update param with delta_param solved by solver.
    virtual void UpdateParam(const Vec &delta_param) = 0;

    // Param operation, backing up and rolling back.
    virtual void BackupParam() = 0;
    virtual void RollbackParam() = 0;

    // If vertex is fixed, solver will not make non-zero increment for this vertex.
    virtual const bool IsFixed() const = 0;
    virtual void SetFixed(bool fixed = true) = 0;

    static uint32_t &global_id() { return global_id_; }

private:
    // Global index for every vertex.
    static uint32_t global_id_;

};

/* Class Vertex Basic Definition. */
template<typename Scalar> uint32_t VertexBasic<Scalar>::global_id_ = 0;

/* Class Vertex declaration. */
template <typename Scalar, int32_t StoreDim, int32_t SolveDim>
class Vertex : public VertexBasic<Scalar> {

public:
    Vertex();
    virtual ~Vertex() = default;

    // Vertex index.
    virtual const uint32_t GetId() const { return id_; }

    // Stored dimension and solve dimension can be different for Quatnion.
    virtual const int32_t GetStoreDimension() const { return StoreDim; }
    virtual const int32_t GetSolveDimension() const { return SolveDim; }

    virtual const std::string GetType();

    // Param operation with scalar adaption.
    virtual void SetParam(const Vec &param) { param_ = param.cast<Scalar>(); }
    virtual Vec GetParam() { return param_.template cast<float>(); }

    // Update param with delta_param solved by solver.
    virtual void UpdateParam(const Vec &delta_param) = 0;

    // Param operation, backing up and rolling back.
    virtual void BackupParam() { param_backup_ = param_; }
    virtual void RollbackParam() { param_ = param_backup_; }

    // If vertex is fixed, solver will not make non-zero increment for this vertex.
    virtual const bool IsFixed() const { return fixed_; }
    virtual void SetFixed(bool fixed = true) { fixed_ = fixed; }

private:
    uint32_t id_ = 0;

    // Store parameter to be solved.
    Eigen::Matrix<Scalar, StoreDim, 1> param_ = Eigen::Matrix<Scalar, StoreDim, 1>::Zero();
    Eigen::Matrix<Scalar, StoreDim, 1> param_backup_ = Eigen::Matrix<Scalar, StoreDim, 1>::Zero();

    // Fix this vertex when solving problem or not.
    bool fixed_ = false;
};

/* Class Vertex Definition. */
template <typename Scalar, int32_t StoreDim, int32_t SolveDim>
Vertex<Scalar, StoreDim, SolveDim>::Vertex() : VertexBasic<Scalar>() {
    id_ = VertexBasic<Scalar>::global_id();
    ++VertexBasic<Scalar>::global_id();
}

template <typename Scalar, int32_t StoreDim, int32_t SolveDim>
const std::string Vertex<Scalar, StoreDim, SolveDim>::GetType() {
    return std::string("Basic Vertex");
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_VERTEX_H_
