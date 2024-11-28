#include "basic_type.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "slam_basic_math.h"
#include "tick_tock.h"

#include "image_painter.h"
#include "visualizor_2d.h"
#include "visualizor_3d.h"

#include "plane.h"
#include "line_segment.h"
#include "general_graph_optimizor.h"
#include "enable_stack_backward.h"

using Scalar = float;
using namespace SLAM_UTILITY;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;

constexpr int32_t kNumberOfCamerasAndLines = 10;

struct Pose {
    Vec3 p_wc = Vec3::Zero();
    Quat q_wc = Quat::Identity();
};

/* Class Edge reprojection. Project orthonormal line (4-dof) on visual norm plane. */
template <typename Scalar>
class EdgeOrthonormalLineToNormPlane : public Edge<Scalar> {
// Vertices are [line, plucker / orthonormal]
//              [camera, p_wc]
//              [camera, q_wc]

public:
    EdgeOrthonormalLineToNormPlane() : Edge<Scalar>(2, 3) {}
    virtual ~EdgeOrthonormalLineToNormPlane() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // TODO:
    }

    virtual void ComputeJacobians() override {
        // TODO:
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().

};

int main(int argc, char **argv) {
    LogFixPercision(6);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with line segment." RESET_COLOR);

    // Generate camera views and line segments.
    std::vector<Pose> cameras_pose;
    std::vector<LineSegment3D> line_segments_3d;
    for (int32_t i = 0; i < kNumberOfCamerasAndLines; ++i) {
        const float theta = i / 2.0f;
        cameras_pose.emplace_back(Pose{
            .p_wc = Vec3(std::cos(theta) * i, i, i * i * 0.1f),
            .q_wc = Quat(Eigen::AngleAxisf(i / 10.0f, Vec3::UnitX())),
        });
        line_segments_3d.emplace_back(LineSegment3D{
            Vec3(- 2.0f * i + std::cos(theta), std::sin(theta), 9 + i),
            Vec3(- i - std::cos(theta), - std::sin(theta), 7 + i + theta)
        });
    }
    ReportDebug("line_segments_3d " << LogVec(line_segments_3d.front().param()));

    // Generate vertex of cameras and lines.
    std::array<std::unique_ptr<Vertex<Scalar>>, kNumberOfCamerasAndLines> all_camera_pos;
    std::array<std::unique_ptr<VertexQuat<Scalar>>, kNumberOfCamerasAndLines> all_camera_rot;
    for (int32_t i = 0; i < kNumberOfCamerasAndLines; ++i) {
        all_camera_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
        all_camera_pos[i]->param() = cameras_pose[i].p_wc;
        all_camera_rot[i] = std::make_unique<VertexQuat<Scalar>>();
        all_camera_rot[i]->param() << cameras_pose[i].q_wc.w(), cameras_pose[i].q_wc.x(),
            cameras_pose[i].q_wc.y(), cameras_pose[i].q_wc.z();
    }
    std::array<std::unique_ptr<Vertex<Scalar>>, kNumberOfCamerasAndLines> all_lines;
    for (int32_t i = 0; i < kNumberOfCamerasAndLines; ++i) {
        all_lines[i] = std::make_unique<Vertex<Scalar>>(4, 4);
        const LineSegment3D noised_line_3d = LineSegment3D(
            line_segments_3d[i].start_point() + Vec3::Random() * 0.2f,
            line_segments_3d[i].end_point() + Vec3::Random() * 0.2f);
        all_lines[i]->param() = LineOrthonormal3D(LinePlucker3D(LineSegment3D(noised_line_3d))).param();
    }

    // Generate edge of observations.
    std::array<std::unique_ptr<EdgeOrthonormalLineToNormPlane<Scalar>>,
        kNumberOfCamerasAndLines * kNumberOfCamerasAndLines> reprojection_edges;
    for (int32_t i = 0; i < kNumberOfCamerasAndLines; ++i) {
        for (int32_t j = 0; j < kNumberOfCamerasAndLines; ++j) {
            const int32_t idx = i * kNumberOfCamerasAndLines + j;
            reprojection_edges[idx] = std::make_unique<EdgeOrthonormalLineToNormPlane<Scalar>>();
            reprojection_edges[idx]->SetVertex(all_lines[j].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[i].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[i].get(), 2);
            // Compute observation.
            const Vec3 p1_c = cameras_pose[i].q_wc.inverse() * (line_segments_3d[j].start_point() - cameras_pose[i].p_wc);
            const Vec3 p2_c = cameras_pose[i].q_wc.inverse() * (line_segments_3d[j].end_point() - cameras_pose[i].p_wc);
            reprojection_edges[idx]->observation() = LineSegment2D(p1_c.head<2>() / p1_c.z(), p2_c.head<2>() / p2_c.z()).param();
            // Add kernel.
            reprojection_edges[idx]->kernel() = std::make_unique<KernelHuber<Scalar>>(0.5f);
            // Do self check.
            reprojection_edges[idx]->SelfCheck();
            reprojection_edges[idx]->SelfCheckJacobians();
        }
    }

    // Construct graph problem and solver.
    Graph<Scalar> problem;
    for (uint32_t i = 0; i < all_camera_pos.size(); ++i) {
        problem.AddVertex(all_camera_pos[i].get());
        problem.AddVertex(all_camera_rot[i].get());
    }
    for (auto &vertex : all_lines) { problem.AddVertex(vertex.get(), false); }
    for (auto &edge : reprojection_edges) { problem.AddEdge(edge.get()); }
    SolverLm<Scalar> solver;
    solver.problem() = &problem;
    TickTock tick_tock;
    solver.Solve(false);
    ReportInfo("[Ticktock] Solve cost time " << tick_tock.TockTickInMillisecond() << " ms");

    // Visualize.
    Visualizor3D::Clear();
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 1.0f,
    });
    // Draw ground truth of camera pose and line in world frame.
    for (const auto &camera_pose : cameras_pose) {
        Visualizor3D::camera_poses().emplace_back(CameraPoseType{
            .p_wc = camera_pose.p_wc,
            .q_wc = camera_pose.q_wc,
            .scale = 0.5f,
        });
    }
    for (const auto &line_segment_3d : line_segments_3d) {
        Visualizor3D::lines().emplace_back(LineType{
            .p_w_i = line_segment_3d.start_point(),
            .p_w_j = line_segment_3d.end_point(),
            .color = RgbColor::kRed,
        });
    }
    // Draw result of optimization.
    for (uint32_t i = 0; i < all_camera_pos.size(); ++i) {
        Quat q_wc = Quat::Identity();
        q_wc.w() = all_camera_rot[i]->param()(0);
        q_wc.x() = all_camera_rot[i]->param()(1);
        q_wc.y() = all_camera_rot[i]->param()(2);
        q_wc.z() = all_camera_rot[i]->param()(3);
        Visualizor3D::camera_poses().emplace_back(CameraPoseType{
            .p_wc = Vec3(all_camera_pos[i]->param()),
            .q_wc = q_wc,
            .scale = 0.3f,
        });
    }
    for (const auto &line : all_lines) {
        const LinePlucker3D plucker(LineOrthonormal3D(line->param()));
        Visualizor3D::lines().emplace_back(LineType{
            .p_w_i = plucker.GetPointOnLine(-1),
            .p_w_j = plucker.GetPointOnLine(1),
            .color = RgbColor::kCyan,
        });
    }

    Visualizor3D::camera_view().p_wc = Vec3(0, 0, -5);
    Visualizor3D::camera_view().q_wc = Quat::Identity();
    while (!Visualizor3D::ShouldQuit()) {
        Visualizor3D::Refresh("Visualizor 3D", 30);
    }

    return 0;
}
