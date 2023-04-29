#ifndef _GENERAL_GRAPH_OPTIMIZOR_VERTEX_H_
#define _GENERAL_GRAPH_OPTIMIZOR_VERTEX_H_

#include "datatype_basic.h"

namespace SLAM_SOLVER {

/* Class Vertex declaration. */
template <typename Scalar>
class Vertex {

public:
    Vertex() = delete;
    Vertex(int32_t param_dim, int32_t delta_dim);
    virtual ~Vertex() = default;

    // Vertex index.
    static uint32_t &GetGlobalId() { return global_id_; }
    const uint32_t GetId() const { return id_; }
    int32_t &ColIndex() { return col_id_; }

    // Param dimension and delta param dimension can be different for Quatnion.
    const int32_t GetParameterDimension() const { return param_dim_; }
    const int32_t GetIncrementDimension() const { return delta_dim_; }

    // Use string to represent vertex type.
    virtual std::string GetType() { return std::string("Basic Vertex"); }

    // Param operation with scalar adaption.
    TVec<Scalar> &Param() { return param_; }

    // Update param with delta_param solved by solver.
    virtual void UpdateParam(const TVec<Scalar> &delta_param) { param_ += delta_param; }

    // Param operation, backing up and rolling back.
    void BackupParam() { param_backup_ = param_; }
    void RollbackParam() { param_ = param_backup_; }

    // If vertex is fixed, solver will not make non-zero increment for this vertex.
    const bool IsFixed() const { return fixed_; }
    void SetFixed(bool fixed = true) { fixed_ = fixed; }

private:
    // Index for every vertex.
    static uint32_t global_id_;
    uint32_t id_ = 0;
    int32_t col_id_ = 0;

    // Param size.
    const int32_t param_dim_;
    const int32_t delta_dim_;

    // Store parameter to be solved.
    TVec<Scalar> param_ = TVec3<Scalar>::Zero();
    TVec<Scalar> param_backup_ = TVec3<Scalar>::Zero();

    // Fix this vertex when solving problem or not.
    bool fixed_ = false;
};

/* Class Vertex Definition. */
template <typename Scalar>
uint32_t Vertex<Scalar>::global_id_ = 0;

template <typename Scalar>
Vertex<Scalar>::Vertex(int32_t param_dim, int32_t delta_dim) : param_dim_(param_dim), delta_dim_(delta_dim) {
    // Resize stored param.
    if (param_.rows() != param_dim_) {
        param_.resize(param_dim_);
        param_backup_.resize(param_dim_);
    }

    // Set index.
    ++Vertex<Scalar>::global_id_;
    id_ = Vertex<Scalar>::global_id_;
}

}

#endif // end of _GENERAL_GRAPH_OPTIMIZOR_VERTEX_H_
