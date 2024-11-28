#include "basic_type.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"
#include "slam_basic_math.h"

#include "general_graph_optimizor.h"
#include "enable_stack_backward.h"

using Scalar = float;
using namespace SLAM_SOLVER;

#include "embeded_generate_sim_data.h"

/* Class Edge reprojection. */
template <typename Scalar>
class EdgeReproject : public Edge<Scalar> {
// vertex is [feature, invdep] [first camera, p_wc0] [first camera, q_wc0] [camera, p_wc] [camera, q_wc].

public:
    EdgeReproject() : Edge<Scalar>(2, 5) {}
    virtual ~EdgeReproject() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        inv_depth0 = this->GetVertex(0)->param()(0);
        p_wc0 = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_i = this->GetVertex(2)->param();
        q_wc0 = TQuat<Scalar>(param_i(0), param_i(1), param_i(2), param_i(3));
        p_wc = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_j = this->GetVertex(4)->param();
        q_wc = TQuat<Scalar>(param_j(0), param_j(1), param_j(2), param_j(3));

        norm_xy0 = this->observation().block(0, 0, 2, 1);
        norm_xy = this->observation().block(2, 0, 2, 1);

        p_c0 = TVec3<Scalar>(norm_xy0(0), norm_xy0(1), static_cast<Scalar>(1)) / inv_depth0;
        p_w = q_wc0 * p_c0 + p_wc0;
        p_c = q_wc.inverse() * (p_w - p_wc);
        inv_depth = static_cast<Scalar>(1) / p_c.z();

        if (std::isinf(inv_depth) || std::isnan(inv_depth)) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c.template head<2>() * inv_depth) - norm_xy;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth) && !std::isnan(inv_depth)) {
            const Scalar inv_depth_2 = inv_depth * inv_depth;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth_2,
                              0, inv_depth, - p_c(1) * inv_depth_2;
        }

        const TMat3<Scalar> R_cw = q_wc.toRotationMatrix().transpose();
        const TMat3<Scalar> R_cc0 = R_cw * q_wc0.matrix();

        const TMat3<Scalar> jacobian_cam0_q = - R_cc0 * Utility::SkewSymmetricMatrix(p_c0);
        const TMat3<Scalar> jacobian_cam0_p = R_cw;

        const TMat3<Scalar> jacobian_cam_q = Utility::SkewSymmetricMatrix(p_c);
        const TMat3<Scalar> jacobian_cam_p = - R_cw;

        const TVec3<Scalar> jacobian_invdep = - R_cc0 *
            TVec3<Scalar>(norm_xy0(0), norm_xy0(1), static_cast<Scalar>(1)) / (inv_depth0 * inv_depth0);

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_cam0_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_cam0_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_cam_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_cam_q;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wc0;
    TQuat<Scalar> q_wc0;
    TVec2<Scalar> norm_xy0;
    Scalar inv_depth0 = 0;
    TVec3<Scalar> p_c0;

    TVec3<Scalar> p_wc;
    TQuat<Scalar> q_wc;
    TVec2<Scalar> norm_xy;
    Scalar inv_depth = 0;
    TVec3<Scalar> p_c;

    TVec3<Scalar> p_w;
};

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with <invdep> <norm plane> model." RESET_COLOR);

    std::vector<Pose<Scalar>> cameras;
    std::vector<TVec3<Scalar>> points;
    GenerateSimulationData(cameras, points);

    // Use include here is interesting.
    #include "embeded_add_invdep_vertices_of_BA.h"

    // Generate edges between cameras and points.
    std::array<std::unique_ptr<EdgeReproject<Scalar>>, (kCameraFrameNumber - 1) * kPointsNumber> reprojection_edges = {};
    int32_t idx = 0;
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        for (int32_t j = 1; j < kCameraFrameNumber; ++j) {
            reprojection_edges[idx] = std::make_unique<EdgeReproject<Scalar>>();
            reprojection_edges[idx]->SetVertex(all_points[i].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[0].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[0].get(), 2);
            reprojection_edges[idx]->SetVertex(all_camera_pos[j].get(), 3);
            reprojection_edges[idx]->SetVertex(all_camera_rot[j].get(), 4);

            TVec3<Scalar> p_c0 = cameras[0].q_wc.inverse() * (points[i] - cameras[0].p_wc);
            TVec3<Scalar> p_cj = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec4<Scalar> obv = TVec4<Scalar>::Zero();
            obv.head<2>() = p_c0.head<2>() / p_c0.z();
            obv.tail<2>() = p_cj.head<2>() / p_cj.z();
            reprojection_edges[idx]->observation() = obv;
            #include "embeded_add_kernel.h"
            reprojection_edges[idx]->SelfCheck();
            reprojection_edges[idx]->SelfCheckJacobians();
            ++idx;
        }
    }

    // Fix first two camera pos.
    all_camera_pos[0]->SetFixed(true);
    all_camera_pos[1]->SetFixed(true);

    // Construct graph problem and solver.
    Graph<Scalar> problem;
    for (uint32_t i = 0; i < all_camera_pos.size(); ++i) {
        problem.AddVertex(all_camera_pos[i].get());
        problem.AddVertex(all_camera_rot[i].get());
    }
    for (auto &vertex : all_points) { problem.AddVertex(vertex.get(), false); }
    for (auto &edge : reprojection_edges) { problem.AddEdge(edge.get()); }

    SolverLm<Scalar> solver;
    solver.problem() = &problem;

    TickTock tick_tock;
    tick_tock.TockTickInMillisecond();
    solver.Solve(false);
    ReportInfo("[Ticktock] Solve cost time " << tick_tock.TockTickInMillisecond() << " ms");

    // Show optimization result.
    #include "embeded_show_optimize_result.h"

    return 0;
}
