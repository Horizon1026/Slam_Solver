#include "basic_type.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"
#include "slam_basic_math.h"

#include "general_graph_optimizor.h"
#include "enable_stack_backward.h"

using Scalar = float;
using namespace SLAM_SOLVER;

#include "embeded_generate_sim_data.h"

/* Class Edge reprojection. Project feature 1-dof invdep on visual norm plane via imu pose. */
template <typename Scalar>
class EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera : public Edge<Scalar> {
// Vertices are [feature, invdep]
//              [first imu pose, p_wi0]
//              [first imu pose, q_wi0]
//              [imu pose, p_wi]
//              [imu pose, q_wi]
//              [extrinsic, p_ic]
//              [extrinsic, q_ic]

public:
    EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera() : Edge<Scalar>(2, 7) {}
    virtual ~EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Extract states.
        inv_depth0_ = this->GetVertex(0)->param()(0);
        p_wi0_ = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_i0 = this->GetVertex(2)->param();
        q_wi0_ = TQuat<Scalar>(param_i0(0), param_i0(1), param_i0(2), param_i0(3));
        p_wi_ = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_i = this->GetVertex(4)->param();
        q_wi_ = TQuat<Scalar>(param_i(0), param_i(1), param_i(2), param_i(3));
        p_ic_ = this->GetVertex(5)->param();
        const TVec4<Scalar> &param_ic = this->GetVertex(6)->param();
        q_ic_ = TQuat<Scalar>(param_ic(0), param_ic(1), param_ic(2), param_ic(3));

        // Extract observations.
        norm_xy0_ = this->observation().block(0, 0, 2, 1);
        norm_xy_ = this->observation().block(2, 0, 2, 1);

        // Compute projection.
        p_c0_ = TVec3<Scalar>(norm_xy0_(0), norm_xy0_(1), static_cast<Scalar>(1)) / inv_depth0_;
        p_i0_ = q_ic_ * p_c0_ + p_ic_;
        p_w_ = q_wi0_ * p_i0_ + p_wi0_;
        p_i_ = q_wi_.inverse() * (p_w_ - p_wi_);
        p_c_ = q_ic_.inverse() * (p_i_ - p_ic_);
        inv_depth_ = static_cast<Scalar>(1) / p_c_.z();

        if (std::isinf(inv_depth_) || std::isnan(inv_depth_) || inv_depth_ < kZero) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c_.template head<2>() * inv_depth_) - norm_xy_;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth_) && !std::isnan(inv_depth_) && inv_depth_ > kZero) {
            const Scalar inv_depth_2 = inv_depth_* inv_depth_;
            jacobian_2d_3d << inv_depth_, 0, - p_c_(0) * inv_depth_2,
                              0, inv_depth_, - p_c_(1) * inv_depth_2;
        }

        const TQuat<Scalar> q_ci = q_ic_.inverse();
        const TQuat<Scalar> q_cw = q_ci * q_wi_.inverse();
        const TQuat<Scalar> q_ci0 = q_cw * q_wi0_;
        const TQuat<Scalar> q_cc0 = q_ci0 * q_ic_;

        const TMat3<Scalar> R_ci = q_ci.toRotationMatrix();
        const TMat3<Scalar> R_cw = q_cw.toRotationMatrix();
        const TMat3<Scalar> R_ci0 = q_ci0.toRotationMatrix();
        const TMat3<Scalar> R_cc0 = q_cc0.toRotationMatrix();

        const TMat3<Scalar> jacobian_cam0_p = R_cw;
        const TMat3<Scalar> jacobian_cam0_q = - R_ci0 * Utility::SkewSymmetricMatrix(p_i0_);

        const TMat3<Scalar> jacobian_cam_p = - R_cw;
        const TMat3<Scalar> jacobian_cam_q = R_ci * Utility::SkewSymmetricMatrix(p_i_);

        const TVec3<Scalar> jacobian_invdep = - R_cc0 *
            TVec3<Scalar>(norm_xy0_.x(), norm_xy0_.y(), static_cast<Scalar>(1)) / (inv_depth0_ * inv_depth0_);

        const TMat3<Scalar> jacobian_ex_p = R_ci * ((q_wi_.inverse() * q_wi0_).matrix() - TMat3<Scalar>::Identity());
        const TMat3<Scalar> jacobian_ex_q = - R_cc0 * Utility::SkewSymmetricMatrix(p_c0_) + Utility::SkewSymmetricMatrix(R_cc0 * p_c0_) +
            Utility::SkewSymmetricMatrix(q_ic_.inverse() * (q_wi_.inverse() * (q_wi0_ * p_ic_ + p_wi0_ - p_wi_) - p_ic_));

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_cam0_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_cam0_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_cam_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_cam_q;
        this->GetJacobian(5) = jacobian_2d_3d * jacobian_ex_p;
        this->GetJacobian(6) = jacobian_2d_3d * jacobian_ex_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc0_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wi0_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi0_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy0_ = TVec2<Scalar>::Zero();
    Scalar inv_depth0_ = 0;
    TVec3<Scalar> p_i0_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c0_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wi_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wi_ = TQuat<Scalar>::Identity();
    TVec2<Scalar> norm_xy_ = TVec2<Scalar>::Zero();
    Scalar inv_depth_ = 0;
    TVec3<Scalar> p_i_ = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_c_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_w_ = TVec3<Scalar>::Zero();

    TVec3<Scalar> p_ic_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_ic_ = TQuat<Scalar>::Identity();
};

/* Class Edge pose prior. This can be used to fix a pose with specified weight. */
template <typename Scalar>
class EdgePriorPose : public Edge<Scalar> {
// Vertices are [position, p_wc]
//              [rotation, q_wc]

public:
    EdgePriorPose() : Edge<Scalar>(6, 2) {}
    virtual ~EdgePriorPose() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        p_wc_ = this->GetVertex(0)->param();
        const TVec4<Scalar> &param = this->GetVertex(1)->param();
        q_wc_ = TQuat<Scalar>(param(0), param(1), param(2), param(3));

        // Get observation.
        obv_p_wc_ = this->observation().block(0, 0, 3, 1);
        const TVec4<Scalar> param_obv = this->observation().block(3, 0, 4, 1);
        obv_q_wc_ = TQuat<Scalar>(param_obv(0), param_obv(1), param_obv(2), param_obv(3));

        // Compute residual.
        this->residual().setZero(6);
        this->residual().head(3) = p_wc_ - obv_p_wc_;
        this->residual().tail(3) = static_cast<Scalar>(2) * (obv_q_wc_.inverse() * q_wc_).vec();
    }

    virtual void ComputeJacobians() override {
        TMat<Scalar> jacobian_p = TMat<Scalar>::Zero(6, 3);
        jacobian_p.block(0, 0, 3, 3).setIdentity();

        TMat<Scalar> jacobian_q = TMat<Scalar>::Zero(6, 3);
        jacobian_q.block(3, 0, 3, 3) = Utility::Qleft(obv_q_wc_.inverse() * q_wc_).template bottomRightCorner<3, 3>();

        this->GetJacobian(0) = jacobian_p;
        this->GetJacobian(1) = jacobian_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_wc_ = TQuat<Scalar>::Identity();

    TVec3<Scalar> obv_p_wc_ = TVec3<Scalar>::Zero();
    TQuat<Scalar> obv_q_wc_ = TQuat<Scalar>::Identity();
};

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with <invdep> <norm plane> model with <T_ic>." RESET_COLOR);

    std::vector<Pose<Scalar>> cameras;
    std::vector<TVec3<Scalar>> points;
    GenerateSimulationData(cameras, points);

    // Use include here is interesting.
    #include "embeded_add_invdep_vertices_of_BA.h"

    // Generate vertex of camera extrinsics.
    TQuat<Scalar> q_ic = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_ic = TVec3<Scalar>::Zero();
    std::array<std::unique_ptr<Vertex<Scalar>>, kCameraExtrinsicNumber> all_camera_ex_pos = {};
    std::array<std::unique_ptr<VertexQuat<Scalar>>, kCameraExtrinsicNumber> all_camera_ex_rot = {};
    for (int32_t i = 0; i < kCameraExtrinsicNumber; ++i) {
        all_camera_ex_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
        all_camera_ex_pos[i]->param() = p_ic;
        all_camera_ex_pos[i]->name() = std::string("p_ic") + std::to_string(i);
        all_camera_ex_rot[i] = std::make_unique<VertexQuat<Scalar>>();
        all_camera_ex_rot[i]->param() << q_ic.w(), q_ic.x(), q_ic.y(), q_ic.z();
        all_camera_ex_rot[i]->name() = std::string("q_ic") + std::to_string(i);
    }

    // Generate edges between cameras and points.
    std::array<std::unique_ptr<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera<Scalar>>, (kCameraFrameNumber - 1) * kPointsNumber> reprojection_edges = {};
    int32_t idx = 0;
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        for (int32_t j = 1; j < kCameraFrameNumber; ++j) {
            reprojection_edges[idx] = std::make_unique<EdgeFeatureInvdepToNormPlaneViaImuWithinTwoFramesOneCamera<Scalar>>();
            reprojection_edges[idx]->SetVertex(all_points[i].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[0].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[0].get(), 2);
            reprojection_edges[idx]->SetVertex(all_camera_pos[j].get(), 3);
            reprojection_edges[idx]->SetVertex(all_camera_rot[j].get(), 4);
            reprojection_edges[idx]->SetVertex(all_camera_ex_pos[0].get(), 5);
            reprojection_edges[idx]->SetVertex(all_camera_ex_rot[0].get(), 6);

            TVec3<Scalar> p_c0 = cameras[0].q_wc.inverse() * (points[i] - cameras[0].p_wc);
            TVec3<Scalar> p_cj = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec4<Scalar> obv = TVec4<Scalar>::Zero();
            obv.head<2>() = p_c0.head<2>() / p_c0.z();
            obv.tail<2>() = p_cj.head<2>() / p_cj.z();
            reprojection_edges[idx]->observation() = obv;
            #include "embeded_add_kernel.h"
            reprojection_edges[idx]->SelfCheck();
            ++idx;
        }
    }

    // Generate prior edges.
    std::array<std::unique_ptr<EdgePriorPose<Scalar>>, kCameraExtrinsicNumber> prior_edges = {};
    prior_edges[0] = std::make_unique<EdgePriorPose<Scalar>>();
    prior_edges[0]->SetVertex(all_camera_ex_pos[0].get(), 0);
    prior_edges[0]->SetVertex(all_camera_ex_rot[0].get(), 1);
    TMat<Scalar> obv = TVec7<Scalar>::Zero();
    obv.block(0, 0, 3, 1) = all_camera_ex_pos[0]->param();
    obv.block(3, 0, 4, 1) = all_camera_ex_rot[0]->param();
    prior_edges[0]->observation() = obv;
    prior_edges[0]->information() = TMat6<Scalar>::Identity() * 1e6;
    prior_edges[0]->SelfCheck();

    // Fix first two camera pos.
    all_camera_pos[0]->SetFixed(true);
    all_camera_pos[1]->SetFixed(true);

    // Construct graph problem and solver.
    Graph<Scalar> problem;
    for (uint32_t i = 0; i < all_camera_ex_pos.size(); ++i) {
        problem.AddVertex(all_camera_ex_pos[i].get());
        problem.AddVertex(all_camera_ex_rot[i].get());
    }
    for (uint32_t i = 0; i < all_camera_pos.size(); ++i) {
        problem.AddVertex(all_camera_pos[i].get());
        problem.AddVertex(all_camera_rot[i].get());
    }
    for (auto &vertex : all_points) { problem.AddVertex(vertex.get(), false); }
    for (auto &edge : reprojection_edges) { problem.AddEdge(edge.get()); }
    for (auto &edge : prior_edges) { problem.AddEdge(edge.get()); }

    SolverLm<Scalar> solver;
    solver.problem() = &problem;

    TickTock tick_tock;
    tick_tock.TockTickInMillisecond();
    solver.Solve(false);
    problem.VerticesInformation();
    ReportInfo("[Ticktock] Solve cost time " << tick_tock.TockTickInMillisecond() << " ms");

    // Show optimization result.
    #include "embeded_show_optimize_result.h"

    return 0;
}
