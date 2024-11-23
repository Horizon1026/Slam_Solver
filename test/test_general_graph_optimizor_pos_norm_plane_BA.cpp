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
// vertex is [feature, p_w] [camera, p_wc] [camera, q_wc].

public:
    EdgeReproject() : Edge<Scalar>(2, 3) {}
    virtual ~EdgeReproject() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        p_w = this->GetVertex(0)->param();
        p_wc = this->GetVertex(1)->param();
        const TVec4<Scalar> &parameter = this->GetVertex(2)->param();
        q_wc = TQuat<Scalar>(parameter(0), parameter(1), parameter(2), parameter(3));

        pixel_norm_xy = this->observation();

        p_c = q_wc.inverse() * (p_w - p_wc);
        inv_depth = static_cast<Scalar>(1) / p_c.z();

        if (std::isinf(inv_depth) || std::isnan(inv_depth)) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c.template head<2>() * inv_depth) - pixel_norm_xy;
        }
    }

    virtual void ComputeJacobians() override {
        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        if (!std::isinf(inv_depth) && !std::isnan(inv_depth)) {
            const Scalar inv_depth_2 = inv_depth * inv_depth;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth_2,
                              0, inv_depth, - p_c(1) * inv_depth_2;
        }

        this->GetJacobian(0) = jacobian_2d_3d * (q_wc.inverse().matrix());
        this->GetJacobian(1) = - this->GetJacobian(0);
        this->GetJacobian(2) = jacobian_2d_3d * SLAM_UTILITY::Utility::SkewSymmetricMatrix(p_c);
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_w;
    TVec3<Scalar> p_wc;
    TQuat<Scalar> q_wc;
    TVec2<Scalar> pixel_norm_xy;
    TVec3<Scalar> p_c;
    Scalar inv_depth = 0;
};

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with <pos> <norm plane> model." RESET_COLOR);

    std::vector<Pose<Scalar>> cameras;
    std::vector<TVec3<Scalar>> points;
    GenerateSimulationData(cameras, points);

    // Use include here is interesting.
    #include "embeded_add_pos_vertices_of_BA.h"

    // Generate edges between cameras and points.
    std::array<std::unique_ptr<EdgeReproject<Scalar>>, kCameraFrameNumber * kPointsNumber> reprojection_edges = {};
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        for (int32_t j = 0; j < kCameraFrameNumber; ++j) {
            const int32_t idx = i * kCameraFrameNumber + j;
            reprojection_edges[idx] = std::make_unique<EdgeReproject<Scalar>>();
            reprojection_edges[idx]->SetVertex(all_points[i].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[j].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[j].get(), 2);

            TVec3<Scalar> p_c = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec2<Scalar> obv = p_c.head<2>() / p_c.z();
            reprojection_edges[idx]->observation() = obv;
            #include "embeded_add_kernel.h"
            reprojection_edges[idx]->SelfCheck();
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
