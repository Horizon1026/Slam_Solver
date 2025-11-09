#include "basic_type.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"

#include "enable_stack_backward.h"
#include "general_graph_optimizor.h"

using Scalar = float;
using namespace slam_solver;

#include "embeded_generate_sim_data.h"

/* Class Edge reprojection. */
template <typename Scalar>
class EdgeReproject: public Edge<Scalar> {
    // vertex is [feature, invdep] [first camera, p_wci] [first camera, q_wci] [camera, p_wc] [camera, q_wc].

public:
    EdgeReproject(): Edge<Scalar>(2, 5) {}
    virtual ~EdgeReproject() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        inv_depth_i = this->GetVertex(0)->param()(0);
        p_wci = this->GetVertex(1)->param();
        const TVec4<Scalar> &param_i = this->GetVertex(2)->param();
        q_wci = TQuat<Scalar>(param_i(0), param_i(1), param_i(2), param_i(3));
        p_wcj = this->GetVertex(3)->param();
        const TVec4<Scalar> &param_j = this->GetVertex(4)->param();
        q_wcj = TQuat<Scalar>(param_j(0), param_j(1), param_j(2), param_j(3));

        norm_xy_i = this->observation().block(0, 0, 2, 1);
        norm_xy_j = this->observation().block(2, 0, 2, 1);

        p_ci = TVec3<Scalar>(norm_xy_i(0), norm_xy_i(1), static_cast<Scalar>(1)) / inv_depth_i;
        p_w = q_wci * p_ci + p_wci;
        p_cj = q_wcj.inverse() * (p_w - p_wcj);
        inv_depth_j = static_cast<Scalar>(1) / p_cj.z();

        const TVec3<Scalar> obv_p_c = TVec3<Scalar>(norm_xy_j.x(), norm_xy_j.y(), 1.0f);
        this->residual() = tangent_base_transpose * (p_cj.normalized() - obv_p_c.normalized());
    }

    virtual void ComputeJacobians() override {
        const Scalar p_c_norm = p_cj.norm();
        const Scalar p_c_norm3 = p_c_norm * p_c_norm * p_c_norm;
        TMat3<Scalar> jacobian_norm = TMat3<Scalar>::Zero();
        if (p_c_norm3 > kZeroFloat) {
            jacobian_norm << 1.0 / p_c_norm - p_cj.x() * p_cj.x() / p_c_norm3, -p_cj.x() * p_cj.y() / p_c_norm3, -p_cj.x() * p_cj.z() / p_c_norm3,
                -p_cj.x() * p_cj.y() / p_c_norm3, 1.0 / p_c_norm - p_cj.y() * p_cj.y() / p_c_norm3, -p_cj.y() * p_cj.z() / p_c_norm3,
                -p_cj.x() * p_cj.z() / p_c_norm3, -p_cj.y() * p_cj.z() / p_c_norm3, 1.0 / p_c_norm - p_cj.z() * p_cj.z() / p_c_norm3;
        }

        TMat2x3<Scalar> jacobian_2d_3d = tangent_base_transpose * jacobian_norm;

        const TMat3<Scalar> jacobian_cami_p = q_wcj.toRotationMatrix().transpose();
        const TMat3<Scalar> jacobian_cami_q = -(q_wcj.inverse() * q_wci).toRotationMatrix() * Utility::SkewSymmetricMatrix(p_ci);

        const TMat3<Scalar> jacobian_camj_p = -jacobian_cami_p;
        const TMat3<Scalar> jacobian_camj_q = Utility::SkewSymmetricMatrix(p_cj);

        const TVec3<Scalar> jacobian_invdep =
            -(q_wcj.inverse() * q_wci).toRotationMatrix() * TVec3<Scalar>(norm_xy_i(0), norm_xy_i(1), static_cast<Scalar>(1)) / (inv_depth_i * inv_depth_i);

        this->GetJacobian(0) = jacobian_2d_3d * jacobian_invdep;
        this->GetJacobian(1) = jacobian_2d_3d * jacobian_cami_p;
        this->GetJacobian(2) = jacobian_2d_3d * jacobian_cami_q;
        this->GetJacobian(3) = jacobian_2d_3d * jacobian_camj_p;
        this->GetJacobian(4) = jacobian_2d_3d * jacobian_camj_q;
    }

    // Set tangent base.
    void SetTrangetBase(const TVec3<Scalar> &vec) { tangent_base_transpose = Utility::TangentBase(vec).transpose(); }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_wci;
    TQuat<Scalar> q_wci;
    TVec2<Scalar> norm_xy_i;
    Scalar inv_depth_i = 0;
    TVec3<Scalar> p_ci;

    TVec3<Scalar> p_wcj;
    TQuat<Scalar> q_wcj;
    TVec2<Scalar> norm_xy_j;
    Scalar inv_depth_j = 0;
    TVec3<Scalar> p_cj;

    TVec3<Scalar> p_w;
    TMat2x3<Scalar> tangent_base_transpose;
};

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with <invdep> <unit sphere> model." RESET_COLOR);

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
            reprojection_edges[idx]->SetTrangetBase(p_c0);
            reprojection_edges[idx]->observation() = obv;
#include "embeded_add_kernel.h"
            reprojection_edges[idx]->SelfCheck();
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
    for (auto &vertex: all_points) {
        problem.AddVertex(vertex.get(), false);
    }
    for (auto &edge: reprojection_edges) {
        problem.AddEdge(edge.get());
    }

    SolverLm<Scalar> solver;
    solver.problem() = &problem;

    TickTock tick_tock;
    tick_tock.TockTickInMillisecond();
    solver.Solve(false);
    ReportInfo("[Ticktock] Solve cost time " << tick_tock.TockTickInMillisecond() << " ms");

// Show optimization result.
#include "embeded_show_optimize_result.h"
    problem.VerticesInformation();
    // problem.EdgesInformation();

    return 0;
}
