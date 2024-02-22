#include "datatype_basic.h"
#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"
#include "visualizor_3d.h"

#include "pose_graph_optimizor.h"
#include "enable_stack_backward.h"

using Scalar = float;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;

/* Simulation Data. */
template <typename Scalar>
struct Pose {
    TVec3<Scalar> p_wb = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wb = TQuat<Scalar>::Identity();
    TMat3<Scalar> cov_p = TMat3<Scalar>::Identity();
    TMat3<Scalar> cov_q = TMat3<Scalar>::Identity();
};

void GenerateSimulationData(std::vector<Pose<Scalar>> &poses) {
    poses.clear();

    // Poses.
    for (int32_t i = 0; i <= 20; ++i) {
        Pose<Scalar> pose;
        const TVec3<Scalar> euler = TVec3<Scalar>(0, -90 + i * 18, 0);
        pose.q_wb = Utility::EulerToQuaternion(euler);
        pose.p_wb = TVec3<Scalar>(0, 0, 0) - pose.q_wb * TVec3<Scalar>(0, 0, 8 + i * 0.1);
        poses.emplace_back(pose);
    }
}

void AddAllRawPosesIntoVisualizor(const std::vector<Pose<Scalar>> &poses) {
    Visualizor3D::Clear();

    // Add word frame.
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 10.0f,
    });
    Visualizor3D::points().emplace_back(PointType{
        .p_w = Vec3::Zero(),
        .color = RgbColor::kWhite,
        .radius = 2,
    });

    // Add all poses.
    for (uint32_t i = 0; i < poses.size(); ++i) {
        Visualizor3D::poses().emplace_back(PoseType{
            .p_wb = poses[i].p_wb,
            .q_wb = poses[i].q_wb,
            .scale = i == 0 ? 2.0f : 1.0f,
        });

        if (i) {
            Visualizor3D::lines().emplace_back(LineType{
                .p_w_i = poses[i - 1].p_wb,
                .p_w_j = poses[i].p_wb,
                .color = RgbColor::kWhite,
            });
        }
    }
}

void AddAllCorrectPosesIntoVisualizor(const std::vector<TVec3<Scalar>> &corr_p_wb,
                                      const std::vector<TQuat<Scalar>> &corr_q_wb) {


    // Add all poses.
    for (uint32_t i = 0; i < corr_p_wb.size(); ++i) {
        Visualizor3D::poses().emplace_back(PoseType{
            .p_wb = corr_p_wb[i],
            .q_wb = corr_q_wb[i],
            .scale = 1.0,
        });

        if (i) {
            Visualizor3D::lines().emplace_back(LineType{
                .p_w_i = corr_p_wb[i - 1],
                .p_w_j = corr_p_wb[i],
                .color = RgbColor::kGreen,
            });
        }
    }
}

void log(const TVec3<Scalar> &p_in, const TQuat<Scalar> &q_in, const Scalar alpha,
         TVec6<Scalar> &v_out) {
    v_out.head<3>() = Utility::Logarithm(q_in);
    v_out.tail<3>() = alpha * p_in;
}

void exp(const TVec6<Scalar> &v_in, const Scalar alpha,
         TVec3<Scalar> &p_out, TQuat<Scalar> &q_out) {
    const TVec3<Scalar> vec = v_in.head<3>();
    q_out = Utility::Exponent(vec);
    p_out = v_in.tail<3>() / alpha;
}

void J(const TVec3<Scalar> &p_A, const TQuat<Scalar> &q_A,
       const TVec3<Scalar> &p_A_, const TQuat<Scalar> &q_A_,
       const Scalar s, const Scalar alpha,
       TVec3<Scalar> &p_out, TQuat<Scalar> &q_out) {
    TVec3<Scalar> p_AinvA_;
    TQuat<Scalar> q_AinvA_;
    Utility::ComputeTransformInverseTransform(p_A, q_A, p_A_, q_A_, p_AinvA_, q_AinvA_);
    TVec6<Scalar> se3;
    log(p_AinvA_, q_AinvA_, alpha, se3);
    se3 = se3 * s;

    TVec3<Scalar> p_exp;
    TQuat<Scalar> q_exp;
    exp(se3, alpha, p_exp, q_exp);

    Utility::ComputeTransformTransform(p_A, q_A, p_exp, q_exp, p_out, q_out);
}

void DoPgoByPoseGraphOptimizor(const std::vector<Pose<Scalar>> &poses,
                               std::vector<TVec3<Scalar>> &corr_p_wb,
                               std::vector<TQuat<Scalar>> &corr_q_wb) {
    corr_p_wb.clear();
    corr_q_wb.clear();
    const uint32_t N = poses.size() - 1;
    ReportInfo("N is " << N);

    // Compute trace of each pose covariance.
    std::vector<Scalar> trace_cov_p;
    std::vector<Scalar> trace_cov_q;
    for (uint32_t i = 0; i < N; ++i) {
        trace_cov_p.emplace_back((poses[i].cov_p + poses[i + 1].cov_p).trace());
        trace_cov_q.emplace_back((poses[i].cov_q + poses[i + 1].cov_q).trace());
    }

    // Compute parameter alpha.
    Scalar numerator = 0;
    Scalar denominator = 0;
    for (uint32_t i = 0; i < N; ++i) {
        numerator += std::sqrt(trace_cov_q[i]);
        denominator += std::sqrt(trace_cov_p[i]);
    }
    const Scalar alpha = numerator / denominator;
    const Scalar alpha_2 = alpha * alpha;

    // Compute summary of covariance trace.
    Scalar sum_of_trace_q = 0;
    Scalar sum_of_trace_p = 0;
    for (uint32_t i = 0; i < N; ++i) {
        sum_of_trace_q += trace_cov_q[i];
        sum_of_trace_p += trace_cov_p[i];
    }

    // Compute weight of each pose.
    std::vector<Scalar> weights;
    denominator = sum_of_trace_q + alpha_2 * sum_of_trace_p;
    for (uint32_t i = 0; i < N; ++i) {
        weights.emplace_back((trace_cov_q[i] + alpha_2 * trace_cov_p[i]) / denominator);
        ReportInfo("Weight " << i << " : " << weights[i]);
    }

    // Compute each relative pose M.
    std::vector<TVec3<Scalar>> all_p_M;
    std::vector<TQuat<Scalar>> all_q_M;
    for (uint32_t i = 0; i < N; ++i) {
        TVec3<Scalar> temp_p_M;
        TQuat<Scalar> temp_q_M;
        Utility::ComputeTransformInverseTransform(poses[i].p_wb, poses[i].q_wb,
            poses[i + 1].p_wb, poses[i + 1].q_wb, temp_p_M, temp_q_M);
        all_p_M.emplace_back(temp_p_M);
        all_q_M.emplace_back(temp_q_M);

        ReportInfo("Mi " << i << " : " << LogVec(all_p_M.back()) << ", " << LogQuat(all_q_M.back()));
    }

    // Compute A and target A(A_).
    TVec3<Scalar> p_A = poses.back().p_wb;
    TQuat<Scalar> q_A = poses.back().q_wb;
    Utility::ComputeTransformInverseTransform(poses.front().p_wb, poses.front().q_wb,
        poses.back().p_wb, poses.back().q_wb, p_A, q_A);
    const TVec3<Scalar> p_A_ = TVec3<Scalar>::Zero();  // This should depend on loop closure pnp.
    const TQuat<Scalar> q_A_ = TQuat<Scalar>::Identity();
    ReportInfo("A " << " : " << LogVec(p_A) << ", " << LogQuat(q_A));
    ReportInfo("A_ " << " : " << LogVec(p_A_) << ", " << LogQuat(q_A_));

    // Compute integration of relative pose M.
    std::vector<TVec3<Scalar>> int_p_M;
    std::vector<TQuat<Scalar>> int_q_M;
    for (uint32_t i = 0; i < all_p_M.size(); ++i) {
        if (i) {
            TVec3<Scalar> temp_int_p_M;
            TQuat<Scalar> temp_int_q_M;
            Utility::ComputeTransformTransform(int_p_M.back(), int_q_M.back(),
                all_p_M[i], all_q_M[i], temp_int_p_M, temp_int_q_M);
            int_p_M.emplace_back(temp_int_p_M);
            int_q_M.emplace_back(temp_int_q_M);
        } else {
            int_p_M.emplace_back(all_p_M[i]);
            int_q_M.emplace_back(all_q_M[i]);
        }
    }
    ReportInfo("M1M2...Mn is " << LogVec(int_p_M.back()) << ", " << LogQuat(int_q_M.back()));

    // Iterate each poses.
    std::vector<TVec3<Scalar>> all_p_U_;
    std::vector<TQuat<Scalar>> all_q_U_;
    for (uint32_t i = 0; i < N; ++i) {
        // Compute Um.
        Scalar weight_left = 0;
        for (uint32_t j = 0; j < i; ++j) {
            weight_left += weights[j];
        }
        TVec3<Scalar> p_U_left;
        TQuat<Scalar> q_U_left;
        J(p_A, q_A, p_A_, q_A_, weight_left, alpha, p_U_left, q_U_left);

        Scalar weight_right = 0;
        for (uint32_t j = 0; j <= i; ++j) {
            weight_right += weights[j];
        }
        TVec3<Scalar> p_U_right;
        TQuat<Scalar> q_U_right;
        J(p_A, q_A, p_A_, q_A_, weight_right, alpha, p_U_right, q_U_right);

        TVec3<Scalar> p_U;
        TQuat<Scalar> q_U;
        Utility::ComputeTransformInverseTransform(p_U_left, q_U_left, p_U_right, q_U_right, p_U, q_U);

        ReportInfo("U " << i << " : " << LogVec(p_U) << ", " << LogQuat(q_U));

        // Compute A_.inv * M1 * M2 *...* Mi.
        TVec3<Scalar> p_A_inv_int_Mi;
        TQuat<Scalar> q_A_inv_int_Mi;
        Utility::ComputeTransformInverseTransform(p_A_, q_A_, int_p_M[i], int_q_M[i],
            p_A_inv_int_Mi, q_A_inv_int_Mi);

        // Compute Um_.
        TVec3<Scalar> p_U__right;
        TQuat<Scalar> q_U__right;
        Utility::ComputeTransformTransform(p_U, q_U, p_A_inv_int_Mi, q_A_inv_int_Mi, p_U__right, q_U__right);
        TVec3<Scalar> p_U_;
        TQuat<Scalar> q_U_;
        Utility::ComputeTransformInverseTransform(p_A_inv_int_Mi, q_A_inv_int_Mi,
            p_U__right, q_U__right, p_U_, q_U_);

        all_p_U_.emplace_back(p_U_);
        all_q_U_.emplace_back(q_U_);

        ReportInfo("U_ " << i << " : " << LogVec(p_U) << ", " << LogQuat(q_U));
    }

    // Correct poses.
    for (uint32_t i = 0; i < poses.size(); ++i) {
        TVec3<Scalar> corr_p;
        TQuat<Scalar> corr_q;
        if (i) {
            TVec3<Scalar> delta_p;
            TQuat<Scalar> delta_q;
            Utility::ComputeTransformTransform(all_p_M[i - 1], all_q_M[i - 1],
                all_p_U_[i - 1], all_q_U_[i - 1], delta_p, delta_q);
            Utility::ComputeTransformTransform(corr_p_wb.back(), corr_q_wb.back(),
                delta_p, delta_q, corr_p, corr_q);
            corr_p_wb.emplace_back(corr_p);
            corr_q_wb.emplace_back(corr_q);
        } else {
            corr_p_wb.emplace_back(poses[0].p_wb);
            corr_q_wb.emplace_back(poses[0].q_wb);
        }
    }
}

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test linear pose graph optimizor." RESET_COLOR);

    std::vector<Pose<Scalar>> poses;
    GenerateSimulationData(poses);
    // Add raw poses for visualizor.
    AddAllRawPosesIntoVisualizor(poses);

    // Do pose graph optimization.
    std::vector<TVec3<Scalar>> corr_p_wb;
    std::vector<TQuat<Scalar>> corr_q_wb;
    DoPgoByPoseGraphOptimizor(poses, corr_p_wb, corr_q_wb);

    // Add correct poses for visualizor.
    AddAllCorrectPosesIntoVisualizor(corr_p_wb, corr_q_wb);

    // Visualize.
    Visualizor3D::camera_view().p_wc = Vec3(0, 0, -20);
    do {
        Visualizor3D::Refresh("Visualizor", 50);
    } while (!Visualizor3D::ShouldQuit());

    return 0;
}
