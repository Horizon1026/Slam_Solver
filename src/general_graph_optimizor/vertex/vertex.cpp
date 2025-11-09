#include "vertex.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class Vertex<float>;
template class Vertex<double>;

// Construnct function.
template <typename Scalar>
Vertex<Scalar>::Vertex(int32_t param_dim, int32_t delta_dim)
    : param_dim_(param_dim)
    , delta_dim_(delta_dim) {
    // Resize stored param.
    if (param_.rows() != param_dim_) {
        param_.resize(param_dim_);
        param_backup_.resize(param_dim_);
    }

    // Set index.
    ++Vertex<Scalar>::global_id_;
    id_ = Vertex<Scalar>::global_id_;
}

}  // namespace SLAM_SOLVER
