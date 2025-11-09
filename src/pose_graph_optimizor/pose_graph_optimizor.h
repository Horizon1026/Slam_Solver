#ifndef _GENERAL_POSE_GRAPH_OPTIMIZOR_H_
#define _GENERAL_POSE_GRAPH_OPTIMIZOR_H_

#include "basic_type.h"
#include "vector"

namespace SLAM_SOLVER {

/* Class Pose Graph Solver Declaration. */
template <typename Scalar>
class PoseGraphOptimizor {

public:
    explicit PoseGraphOptimizor() = default;
    virtual ~PoseGraphOptimizor() = default;

    bool Solve();
    void Reset();

    // Reference for member variables.
    std::vector<TVec3<Scalar>> &all_p_wb() { return all_p_wb_; }
    std::vector<TQuat<Scalar>> &all_q_wb() { return all_q_wb_; }
    std::vector<TMat3<Scalar>> &all_cov_p() { return all_cov_p_; }
    std::vector<TMat3<Scalar>> &all_cov_q() { return all_cov_q_; }
    TVec3<Scalar> &desired_p_wb() { return desired_p_wb_; }
    TQuat<Scalar> &desired_q_wb() { return desired_q_wb_; }

    // Const Reference for member variables.
    const std::vector<TVec3<Scalar>> &all_p_wb() const { return all_p_wb_; }
    const std::vector<TQuat<Scalar>> &all_q_wb() const { return all_q_wb_; }
    const std::vector<TMat3<Scalar>> &all_cov_p() const { return all_cov_p_; }
    const std::vector<TMat3<Scalar>> &all_cov_q() const { return all_cov_q_; }
    const TVec3<Scalar> &desired_p_wb() const { return desired_p_wb_; }
    const TQuat<Scalar> &desired_q_wb() const { return desired_q_wb_; }

private:
    void LogMap(const TVec3<Scalar> &p_in, const TQuat<Scalar> &q_in, TVec6<Scalar> &v_out);
    void ExpMap(const TVec6<Scalar> &v_in, TVec3<Scalar> &p_out, TQuat<Scalar> &q_out);
    void FunctionJ(const Scalar s, TVec3<Scalar> &p_out, TQuat<Scalar> &q_out);

    void ComputeAlphaAndWeights();
    void ComputeRelativePoses();
    void ComputeDeltaRelativePoses();
    void CorrectAllPoses();

private:
    // Input member variables.
    std::vector<TVec3<Scalar>> all_p_wb_;
    std::vector<TQuat<Scalar>> all_q_wb_;
    std::vector<TMat3<Scalar>> all_cov_p_;
    std::vector<TMat3<Scalar>> all_cov_q_;
    TVec3<Scalar> desired_p_wb_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> desired_q_wb_ = TQuat<Scalar>::Identity();

    // Template member variables.
    Scalar alpha_ = 0;
    TVec3<Scalar> p_raw_A_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_raw_A_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_target_A_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_target_A_ = TQuat<Scalar>::Identity();

    std::vector<Scalar> weights_;
    std::vector<TVec3<Scalar>> all_p_M_;
    std::vector<TQuat<Scalar>> all_q_M_;
    std::vector<TVec3<Scalar>> all_p_U_;
    std::vector<TQuat<Scalar>> all_q_U_;
};

}  // namespace SLAM_SOLVER

#endif  // end of _GENERAL_POSE_GRAPH_OPTIMIZOR_H_
