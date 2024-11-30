#include "edge.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class Edge<float>;
template class Edge<double>;

// Construct function.
template <typename Scalar>
Edge<Scalar>::Edge(int32_t residual_dim, int32_t vertex_num) {
    // Resize size of residual and jacobian.
    if (residual_.size() != residual_dim) {
        residual_.resize(residual_dim);
        information_.setIdentity(residual_dim, residual_dim);
    }
    vertices_.resize(vertex_num, nullptr);
    jacobians_.resize(vertex_num);

    // Set index.
    ++Edge<Scalar>::global_id_;
    id_ = Edge<Scalar>::global_id_;
}

// Set vertex of this edge.
template <typename Scalar>
bool Edge<Scalar>::SetVertex(Vertex<Scalar> * vertex, uint32_t index) {
    if (vertex == nullptr || index > vertices_.size() - 1) {
        return false;
    }

    vertices_[index] = vertex;
    jacobians_[index].setZero(residual_.rows(), vertices_[index]->GetIncrementDimension());
    return true;
}

template <typename Scalar>
bool Edge<Scalar>::SetVertices(const std::vector<Vertex<Scalar> *> &vertices) {
    if (vertices_.size() != vertices.size()) {
        return false;
    }

    for (const auto &vertex : vertices) {
        if (vertex == nullptr) {
            return false;
        }
    }

    // Copy vertices and resize jacobians.
    for (uint32_t i = 0; i < vertices_.size(); ++i) {
        vertices_[i] = vertices[i];
        jacobians_[i].setZero(residual_.rows(), vertices_[i]->GetIncrementDimension());
    }
    return true;
}

// Check if this edge valid.
template <typename Scalar>
bool Edge<Scalar>::SelfCheck() {
    if (vertices_.size() != jacobians_.size()) {
        ReportError("[Edge] Vertices size is not equal to jacobians size.");
        return false;
    }

    if (residual_.rows() != information_.rows() || residual_.rows() != information_.cols()) {
        ReportError("[Edge] Residual size doesn't match information matrix.");
        return false;
    }

    for (uint32_t i = 0; i < vertices_.size(); ++i) {
        if (vertices_[i] == nullptr) {
            ReportError("[Edge] Vertex " << i << " is nullptr.");
            return false;
        }

        if (vertices_[i]->GetIncrementDimension() != jacobians_[i].cols()) {
            ReportError("[Edge] Vertex " << i << " doesn't match its jacobian.");
            return false;
        }

        if (residual_.rows() != jacobians_[i].rows()) {
            ReportError("[Edge] Residual size doesn't match jacobian.");
            return false;
        }
    }

    if (kernel_ == nullptr) {
        ReportError("[Edge] Kernel function is nullptr.");
        return false;
    }

    return true;
}

template <typename Scalar>
bool Edge<Scalar>::SelfCheckJacobians() {
    const Scalar disturb_step = static_cast<Scalar>(1e-3);

    // Compute residual and jacobian at linearized point.
    ComputeResidual();
    ComputeJacobians();
    const TVec<Scalar> residual = residual_;

    // Compute residual at disturbance based on linearized point.
    std::vector<TVec<Scalar>> linearized_residuals;
    for (const auto &jacobian : jacobians_) {
        linearized_residuals.emplace_back(residual + jacobian * TVec<Scalar>::Ones(jacobian.cols()) * disturb_step);
    }

    // Compute residual at new linearized point.
    std::vector<TVec<Scalar>> directly_residuals;
    for (auto &vertex : vertices_) {
        const TVec<Scalar> delta_param = TVec<Scalar>::Ones(vertex->GetIncrementDimension()) * disturb_step;
        vertex->BackupParam();
        vertex->UpdateParam(delta_param);
        ComputeResidual();
        directly_residuals.emplace_back(residual_);
        vertex->RollbackParam();
    }

    // Compare difference of residuals.
    bool is_jacobian_error = false;
    for (uint32_t i = 0; i < linearized_residuals.size(); ++i) {
        const auto residual = linearized_residuals[i] - directly_residuals[i];
        if (residual.norm() > static_cast<Scalar>(1e-3)) {
            is_jacobian_error = true;
            ReportError("[Edge] Edge <" << name_ << "> self check jacobians error, residual " << i << " is over large : " << LogVec(residual));
        }
    }

    return is_jacobian_error;
}

template <typename Scalar>
void Edge<Scalar>::ComputeNumbericalJacobians() {
    const Scalar eps = 1e-6;
    for (uint32_t i = 0; i < vertices_.size(); ++i) {
        auto &vertex = vertices_[i];
        ComputeResidual();
        const TVec<Scalar> residual = this->residual();
        vertex->BackupParam();
        for (int32_t j = 0; j < vertex->GetIncrementDimension(); ++j) {
            TVec<Scalar> dx = TVec<Scalar>::Zero(vertex->GetIncrementDimension());
            dx(j) = eps;
            vertex->RollbackParam();
            vertex->UpdateParam(dx);
            ComputeResidual();
            jacobians_[i].col(j) = (this->residual() - residual) / eps;
        }
    }
}

}
