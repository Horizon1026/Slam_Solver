#include "basic_type.h"
#include "slam_log_reporter.h"
#include "slam_operations.h"
#include "slam_basic_math.h"
#include "tick_tock.h"
#include "visualizor_3d.h"

#include "plane.h"
#include "line_segment.h"
#include "general_graph_optimizor.h"
#include "enable_stack_backward.h"

using Scalar = float;
using namespace SLAM_UTILITY;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;

constexpr int32_t kNumberOfCamerasAndLines = 10;

struct Pose {
    Vec3 p_wc = Vec3::Zero();
    Quat q_wc = Quat::Identity();
};

/* Class Edge reprojection. Project orthonormal line (4-dof) on visual norm plane. */
template <typename Scalar>
class EdgeOrthonormalLineToNormPlane : public Edge<Scalar> {
// Vertices are [orthonormal line, [theta_x, theta_y, theta_z, phi]]
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

};

int main(int argc, char **argv) {
    LogFixPercision(3);
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
            Vec3(- 2.0f * i + std::cos(theta), std::sin(theta), 5 + i),
            Vec3(- i - std::cos(theta), - std::sin(theta), 2 + i + theta)
        });
    }

    // Generate vertex of cameras and lines.
    std::array<std::unique_ptr<Vertex<Scalar>>, kNumberOfCamerasAndLines> all_camera_pos;
    std::array<std::unique_ptr<VertexQuat<Scalar>>, kNumberOfCamerasAndLines> all_camera_rot;
    for (int32_t i = 0; i < kNumberOfCamerasAndLines; ++i) {
        all_camera_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
        all_camera_pos[i]->param() = cameras_pose[i].p_wc.cast<Scalar>();
        all_camera_rot[i] = std::make_unique<VertexQuat<Scalar>>(4, 3);
        all_camera_rot[i]->param() << cameras_pose[i].q_wc.w(), cameras_pose[i].q_wc.x(),
            cameras_pose[i].q_wc.y(), cameras_pose[i].q_wc.z();
    }
    std::array<std::unique_ptr<Vertex<Scalar>>, kNumberOfCamerasAndLines> all_lines;
    for (int32_t i = 0; i < kNumberOfCamerasAndLines; ++i) {
        all_lines[i] = std::make_unique<Vertex<Scalar>>(4, 4);
        all_lines[i]->param() = LineOrthonormal3D(LinePlucker3D(LineSegment3D(line_segments_3d[i]))).param().cast<Scalar>();
    }

    // TODO:
    // Generate edge of observations.

    // Visualize.
    Visualizor3D::Clear();
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 1.0f,
    });
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

    Visualizor3D::camera_view().p_wc = Vec3(0, 0, -5);
    Visualizor3D::camera_view().q_wc = Quat::Identity();
    while (!Visualizor3D::ShouldQuit()) {
        Visualizor3D::Refresh("Visualizor 3D", 30);
    }

    return 0;
}
