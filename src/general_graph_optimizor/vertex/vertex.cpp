#include "vertex.h"

namespace SLAM_SOLVER {

// Construnct function.
template Vertex<float>::Vertex(int32_t param_dim, int32_t delta_dim);
template Vertex<double>::Vertex(int32_t param_dim, int32_t delta_dim);
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
