#include "pose_graph_optimizor.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"

namespace SLAM_SOLVER {

/* Specialized Template Class Declaration. */
template class PoseGraphOptimizor<float>;
template class PoseGraphOptimizor<double>;

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::Reset() {
    all_p_wb_.clear();
    all_q_wb_.clear();
    all_cov_p_.clear();
    all_cov_q_.clear();

    weights_.clear();

    all_p_M_.clear();
    all_q_M_.clear();
    all_p_U_.clear();
    all_q_U_.clear();
}

template <typename Scalar>
bool PoseGraphOptimizor<Scalar>::Solve() {
    RETURN_FALSE_IF(all_q_wb_.size() != all_p_wb_.size());

    // Compute alpha and weights.
    ComputeAlphaAndWeights();
    // Compute relative poses.
    ComputeRelativePoses();
    // Compute delta relative poses.
    ComputeDeltaRelativePoses();
    // Correct all poses.
    CorrectAllPoses();

    return true;
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::ComputeAlphaAndWeights() {
    // If covariance is not valid, directly compute alpha and weights.
    if (all_p_wb_.size() != all_cov_p_.size() || all_q_wb_.size() != all_cov_q_.size()) {
        alpha_ = static_cast<Scalar>(0.5);
        weights_.resize(all_p_wb_.size() - 1, static_cast<Scalar>(1) / static_cast<Scalar>(all_p_wb_.size() - 1));
        return;
    }

    // Compute trace of each covariance.
    const uint32_t size = all_p_wb_.size() - 1;
    std::vector<Scalar> all_trace_cov_p;
    std::vector<Scalar> all_trace_cov_q;
    all_trace_cov_p.reserve(size);
    all_trace_cov_q.reserve(size);
    for (uint32_t i = 1; i < all_p_wb_.size(); ++i) {
        const Scalar trace_p = (all_cov_p_[i - 1] + all_cov_p_[i]).trace();
        const Scalar trace_q = (all_cov_q_[i - 1] + all_cov_q_[i]).trace();
        all_trace_cov_p.emplace_back(trace_p);
        all_trace_cov_q.emplace_back(trace_q);
    }

    // Compute summary of covariance trace.
    Scalar sum_of_trace_p = 0;
    Scalar sum_of_trace_q = 0;
    for (uint32_t i = 0; i < all_trace_cov_p.size(); ++i) {
        sum_of_trace_p += all_trace_cov_p[i];
        sum_of_trace_q += all_trace_cov_q[i];
    }

    // Compute alpha.
    Scalar numerator = 0;
    Scalar denominator = 0;
    for (uint32_t i = 0; i < all_trace_cov_p.size(); ++i) {
        numerator += std::sqrt(all_trace_cov_p[i]);
        denominator += std::sqrt(all_trace_cov_q[i]);
    }
    alpha_ = numerator / denominator;

    // Compute weights.
    const Scalar alpha_2 = alpha_ * alpha_;
    const Scalar scale = sum_of_trace_q + alpha_2 * sum_of_trace_p;
    for (uint32_t i = 0; i < all_trace_cov_p.size(); ++i) {
        weights_.emplace_back((all_trace_cov_q[i] + alpha_2 * all_trace_cov_p[i]) / scale);
    }
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::ComputeRelativePoses() {
    // Compute raw A and target A.
    Utility::ComputeTransformInverseTransform(all_p_wb_.front(), all_q_wb_.front(),
        all_p_wb_.back(), all_q_wb_.back(), p_raw_A_, q_raw_A_);
    Utility::ComputeTransformInverseTransform(all_p_wb_.front(), all_q_wb_.front(),
        desired_p_wb_, desired_q_wb_, p_target_A_, q_target_A_);

    // Compute each relative poses.
    const uint32_t size = all_p_wb_.size() - 1;
    all_p_M_.reserve(size);
    all_q_M_.reserve(size);
    for (uint32_t i = 1; i < all_p_wb_.size(); ++i) {
        TVec3<Scalar> p_M = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_M = TQuat<Scalar>::Identity();
        Utility::ComputeTransformInverseTransform(all_p_wb_[i - 1], all_q_wb_[i - 1],
            all_p_wb_[i], all_q_wb_[i], p_M, q_M);
        all_p_M_.emplace_back(p_M);
        all_q_M_.emplace_back(q_M);
    }
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::LogMap(const TVec3<Scalar> &p_in,
                                        const TQuat<Scalar> &q_in,
                                        TVec6<Scalar> &v_out) {
    v_out.template head<3>() = Utility::Logarithm(q_in);
    v_out.template tail<3>() = alpha_ * p_in;
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::ExpMap(const TVec6<Scalar> &v_in,
                                        TVec3<Scalar> &p_out,
                                        TQuat<Scalar> &q_out) {
    const TVec3<Scalar> vec = v_in.template head<3>();
    q_out = Utility::Exponent(vec);
    p_out = v_in.template tail<3>() / alpha_;
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::FunctionJ(const Scalar s,
                                           TVec3<Scalar> &p_out,
                                           TQuat<Scalar> &q_out) {
    TVec3<Scalar> temp_p = TVec3<Scalar>::Zero();
    TQuat<Scalar> temp_q = TQuat<Scalar>::Identity();
    Utility::ComputeTransformInverseTransform(p_raw_A_, q_raw_A_, p_target_A_, q_target_A_, temp_p, temp_q);

    TVec6<Scalar> se3 = TVec6<Scalar>::Zero();
    LogMap(temp_p, temp_q, se3);
    se3 = se3 * s;

    ExpMap(se3, temp_p, temp_q);
    Utility::ComputeTransformTransform(p_raw_A_, q_raw_A_, temp_p, temp_q, p_out, q_out);
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::ComputeDeltaRelativePoses() {
    Scalar integrate_weight = static_cast<Scalar>(0);
    TVec3<Scalar> integrate_p_M = TVec3<Scalar>::Zero();
    TQuat<Scalar> integrate_q_M = TQuat<Scalar>::Identity();

    for (uint32_t i = 0; i < all_p_M_.size(); ++i) {
        // Compute Ui in A_ = M1M2...Mn * U1U2...Un
        TVec3<Scalar> p_U_left = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_U_left = TQuat<Scalar>::Identity();
        FunctionJ(integrate_weight, p_U_left, q_U_left);

        TVec3<Scalar> p_U_right = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_U_right = TQuat<Scalar>::Identity();
        integrate_weight += weights_[i];
        FunctionJ(integrate_weight, p_U_right, q_U_right);

        TVec3<Scalar> p_U = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_U = TQuat<Scalar>::Identity();
        Utility::ComputeTransformInverseTransform(p_U_left, q_U_left, p_U_right, q_U_right, p_U, q_U);

        // Compute M1 * M2 *...* Mi.
        TVec3<Scalar> prev_integrate_p_M = integrate_p_M;
        TQuat<Scalar> prev_integrate_q_M = integrate_q_M;
        Utility::ComputeTransformTransform(prev_integrate_p_M, prev_integrate_q_M,
            all_p_M_[i], all_q_M_[i], integrate_p_M, integrate_q_M);

        // Compute A_.inv * M1 * M2 *...* Mi.
        TVec3<Scalar> p_A_inv_int_Mi = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_A_inv_int_Mi = TQuat<Scalar>::Identity();
        Utility::ComputeTransformInverseTransform(p_target_A_, q_target_A_, integrate_p_M, integrate_q_M,
            p_A_inv_int_Mi, q_A_inv_int_Mi);

        // Compute U_i in A_ = M1U1 * M2U2 * ... * MnUn.
        TVec3<Scalar> p_U__right = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_U__right = TQuat<Scalar>::Identity();
        Utility::ComputeTransformTransform(p_U, q_U, p_A_inv_int_Mi, q_A_inv_int_Mi, p_U__right, q_U__right);
        TVec3<Scalar> p_U_ = TVec3<Scalar>::Zero();
        TQuat<Scalar> q_U_ = TQuat<Scalar>::Identity();
        Utility::ComputeTransformInverseTransform(p_A_inv_int_Mi, q_A_inv_int_Mi, p_U__right, q_U__right, p_U_, q_U_);

        all_p_U_.emplace_back(p_U_);
        all_q_U_.emplace_back(q_U_);
    }
}

template <typename Scalar>
void PoseGraphOptimizor<Scalar>::CorrectAllPoses() {
    for (uint32_t i = 0; i < all_p_M_.size(); ++i) {
        TVec3<Scalar> delta_p = TVec3<Scalar>::Zero();
        TQuat<Scalar> delta_q = TQuat<Scalar>::Identity();
        Utility::ComputeTransformTransform(all_p_M_[i], all_q_M_[i], all_p_U_[i], all_q_U_[i], delta_p, delta_q);

        TVec3<Scalar> corr_p = TVec3<Scalar>::Zero();
        TQuat<Scalar> corr_q = TQuat<Scalar>::Identity();
        Utility::ComputeTransformTransform(all_p_wb_[i], all_q_wb_[i], delta_p, delta_q, corr_p, corr_q);
        all_p_wb_[i + 1] = corr_p;
        all_q_wb_[i + 1] = corr_q;
    }
}

}
