#include "basic_type.h"
#include "slam_basic_math.h"
#include "slam_log_reporter.h"
#include "tick_tock.h"

#include "enable_stack_backward.h"
#include "general_graph_optimizor.h"
#include "edge/edge_auto_diff.h"
#include "visualizor_3d.h"

using Scalar = float;
using namespace slam_solver;
using namespace slam_visualizor;

#include "embeded_generate_sim_data.h"

template <typename Scalar>
struct ReprojectCostFunction : public CostFunction<Scalar, 2, 1, 3, 4, 3, 4> {
    ReprojectCostFunction(const TVec4<Scalar>& observation) : observation_(observation) {}

    template <typename T>
    bool Evaluate(const T* const* parameters, T* residuals) const {
        const T& inv_depth0 = parameters[0][0];
        const T* p_wc0_ptr = parameters[1];
        const T* q_wc0_ptr = parameters[2];
        const T* p_wc_ptr = parameters[3];
        const T* q_wc_ptr = parameters[4];

        Eigen::Map<const TVec3<T>> p_wc0(p_wc0_ptr);
        Eigen::Map<const TQuat<T>> q_wc0(q_wc0_ptr);
        Eigen::Map<const TVec3<T>> p_wc(p_wc_ptr);
        Eigen::Map<const TQuat<T>> q_wc(q_wc_ptr);

        TVec2<T> norm_xy0 = observation_.template head<2>().template cast<T>();
        TVec2<T> norm_xy = observation_.template tail<2>().template cast<T>();

        TVec3<T> p_c0 = TVec3<T>(norm_xy0(0), norm_xy0(1), T(1.0)) / inv_depth0;
        TVec3<T> p_w = q_wc0 * p_c0 + p_wc0;
        TVec3<T> p_c = q_wc.inverse() * (p_w - p_wc);
        T inv_depth = T(1.0) / p_c.z();

        residuals[0] = p_c.x() * inv_depth - norm_xy(0);
        residuals[1] = p_c.y() * inv_depth - norm_xy(1);

        return true;
    }

    TVec4<Scalar> observation_;
};

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with <invdep> <norm plane> model (Auto-Diff)." RESET_COLOR);

    std::vector<Pose<Scalar>> cameras;
    std::vector<TVec3<Scalar>> points;
    GenerateSimulationData(cameras, points);

#include "embeded_add_invdep_vertices_of_BA.h"

    // Generate edges between cameras and points.
    std::vector<std::unique_ptr<EdgeAutoDiff<ReprojectCostFunction<Scalar>, Scalar, 2, 1, 3, 4, 3, 4>>> reprojection_edges;
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        for (int32_t j = 1; j < kCameraFrameNumber; ++j) {
            TVec3<Scalar> p_c0 = cameras[0].q_wc.inverse() * (points[i] - cameras[0].p_wc);
            TVec3<Scalar> p_cj = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec4<Scalar> obv = TVec4<Scalar>::Zero();
            obv.head<2>() = p_c0.head<2>() / p_c0.z();
            obv.tail<2>() = p_cj.head<2>() / p_cj.z();

            auto functor = new ReprojectCostFunction<Scalar>(obv);
            auto edge = std::make_unique<EdgeAutoDiff<ReprojectCostFunction<Scalar>, Scalar, 2, 1, 3, 4, 3, 4>>(functor);
            edge->SetVertex(all_points[i].get(), 0);
            edge->SetVertex(all_camera_pos[0].get(), 1);
            edge->SetVertex(all_camera_rot[0].get(), 2);
            edge->SetVertex(all_camera_pos[j].get(), 3);
            edge->SetVertex(all_camera_rot[j].get(), 4);
            edge->observation() = obv;

            edge->kernel() = std::make_unique<KernelHuber<Scalar>>(0.5f);
            edge->SelfCheck();
            edge->SelfCheckJacobians();
            reprojection_edges.emplace_back(std::move(edge));
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

#include "embeded_show_optimize_result.h"

    return 0;
}
