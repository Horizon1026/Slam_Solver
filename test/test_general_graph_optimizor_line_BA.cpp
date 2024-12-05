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

using Scalar = double;
using namespace SLAM_UTILITY;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;
using namespace IMAGE_PAINTER;

constexpr int32_t kNumberOfCameras = 4;
constexpr int32_t kNumberOfLines = 4;

struct Pose {
    Vec3 p_wc = Vec3::Zero();
    Quat q_wc = Quat::Identity();
};

/* Class Vertex Quaternion declaration. */
template <typename Scalar>
class VertexLine : public Vertex<Scalar> {

public:
    VertexLine() : Vertex<Scalar>(6, 4) {}
    virtual ~VertexLine() = default;

    // Update param with delta_param solved by solver.
    virtual void UpdateParam(const TVec<Scalar> &delta_param) override {
        LinePlucker3D plucker_w(Vec6(this->param().template cast<float>()));
        plucker_w.UpdateParameters<true>(Vec4(delta_param.template cast<float>()));
        this->param() = plucker_w.param().cast<Scalar>();
    }
};

/* Class Edge reprojection. Project orthonormal line (4-dof) on visual norm plane. */
template <typename Scalar>
class EdgeOrthonormalLineToNormPlane : public Edge<Scalar> {
// Vertices are [line, plucker]
//              [camera, p_wc]
//              [camera, q_wc]

public:
    EdgeOrthonormalLineToNormPlane() : Edge<Scalar>(2, 3) {}
    virtual ~EdgeOrthonormalLineToNormPlane() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        // Extract parameters.
        const LinePlucker3D plucker_w(Vec6(this->GetVertex(0)->param().template cast<float>()));
        n_w = plucker_w.normal_vector().template cast<Scalar>();
        d_w = plucker_w.direction_vector().template cast<Scalar>();
        jacobian_plucker_to_orthonormal = plucker_w.LinearizeTo4Dof<true>().cast<Scalar>();
        p_wc = this->GetVertex(1)->param().template cast<Scalar>();
        const TVec4<Scalar> &parameter = this->GetVertex(2)->param();
        const TQuat<Scalar> q_wc = TQuat<Scalar>(parameter(0), parameter(1), parameter(2), parameter(3));
        R_wc = TMat3<Scalar>(q_wc);
        // Extract observation.
        const LineSegment2D line_2d(this->observation().template cast<float>());
        s_point = line_2d.start_point_homogeneous().cast<Scalar>();
        e_point = line_2d.end_point_homogeneous().cast<Scalar>();
        // Do reprojection.
        const LinePlucker3D plucker_c = plucker_w.TransformTo(p_wc.template cast<float>(), q_wc.template cast<float>());
        l = plucker_c.ProjectToNormalPlane().cast<Scalar>();
        const Scalar l_2_2 = l(0) * l(0) + l(1) * l(1);
        const Scalar l_1_2 = std::sqrt(l_2_2);
        // Compute residual. Define it by distance from point to line.
        this->residual() = TVec2<Scalar>(s_point.dot(l) / l_1_2,
                                         e_point.dot(l) / l_1_2);
    }

    // TODO: Something wrong here.
    virtual void ComputeJacobians() override {
        const Scalar l_2_2 = l(0) * l(0) + l(1) * l(1);
        const Scalar l_1_2 = std::sqrt(l_2_2);
        const Scalar l_3_2 = l_2_2 * l_1_2;
        const TMat3<Scalar> R_cw = R_wc.transpose();
        const TVec3<Scalar> p_cw(- R_cw * p_wc);

        // Compute jacobian of d_residual to d_line_in_c.
        TMat2x3<Scalar> jacobian_residual_line_in_c = TMat2x3<Scalar>::Zero();
        jacobian_residual_line_in_c << s_point.x() / l_1_2 - l[0] * s_point.dot(l) / l_3_2,
                                       s_point.y() / l_1_2 - l[1] * s_point.dot(l) / l_3_2,
                                       1.0f / l_1_2,
                                       e_point.x() / l_1_2 - l[0] * e_point.dot(l) / l_3_2,
                                       e_point.y() / l_1_2 - l[1] * e_point.dot(l) / l_3_2,
                                       1.0f / l_1_2;
        // Compute jacobian of d_line_in_c to d_plucker_in_c.
        TMat3x6<Scalar> jacobian_line_to_plucker = TMat3x6<Scalar>::Zero();
        jacobian_line_to_plucker.template block<3, 3>(0, 0) = TMat3<Scalar>::Identity();
        // Compute jacobian of d_plucker_in_c to d_plucker_in_w.
        TMat6<Scalar> jacobian_plucker_c_to_w = TMat6<Scalar>::Zero();
        jacobian_plucker_c_to_w.template block<3, 3>(0, 0) = R_cw;
        jacobian_plucker_c_to_w.template block<3, 3>(0, 3) = Utility::SkewSymmetricMatrix(p_cw) * R_cw;
        jacobian_plucker_c_to_w.template block<3, 3>(3, 3) = R_cw;
        // Compute jacobian of d_plucker_in_c to d_camera_pos.
        TMat6x3<Scalar> jacobian_plucker_to_camera_pos = TMat6x3<Scalar>::Zero();
        jacobian_plucker_to_camera_pos.template block<3, 3>(0, 0) = R_cw * Utility::SkewSymmetricMatrix(d_w);
        // TODO: Compute jacobian of d_plucker_in_c to d_camera_rot.
        TMat6x3<Scalar> jacobian_plucker_to_camera_rot = TMat6x3<Scalar>::Zero();
        jacobian_plucker_to_camera_rot.template block<3, 3>(0, 0) = Utility::SkewSymmetricMatrix(R_cw * n_w)
            - Utility::SkewSymmetricMatrix(R_cw * Utility::SkewSymmetricMatrix(p_wc) * d_w);
        jacobian_plucker_to_camera_rot.template block<3, 3>(3, 0) = Utility::SkewSymmetricMatrix(R_cw * d_w);

        // Set jacobian of d_residual to d_line.
        this->GetJacobian(0) = jacobian_residual_line_in_c *
                               jacobian_line_to_plucker *
                               jacobian_plucker_c_to_w *
                               jacobian_plucker_to_orthonormal;
        // Set jacobian of d_residual to d_camera_pose.
        this->GetJacobian(1) = jacobian_residual_line_in_c *
                               jacobian_line_to_plucker *
                               jacobian_plucker_to_camera_pos;
        this->GetJacobian(2) = jacobian_residual_line_in_c *
                               jacobian_line_to_plucker *
                               jacobian_plucker_to_camera_rot;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> n_w = TVec3<Scalar>::Zero();
    TVec3<Scalar> d_w = TVec3<Scalar>::Zero();
    TVec3<Scalar> p_wc = TVec3<Scalar>::Zero();
    TMat3<Scalar> R_wc = TMat3<Scalar>::Identity();
    TVec3<Scalar> l = TVec3<Scalar>::Zero();
    TVec3<Scalar> s_point = TVec3<Scalar>::Zero();
    TVec3<Scalar> e_point = TVec3<Scalar>::Zero();
    TMat6x4<Scalar> jacobian_plucker_to_orthonormal = TMat6x4<Scalar>::Zero();
};

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with line segment." RESET_COLOR);

    // Generate camera views and line segments.
    std::vector<Pose> cameras_pose;
    const float radius = 8.0f;
    for (int32_t i = 0; i < kNumberOfCameras; ++i) {
        const float theta = i * 2 * kPai / (kNumberOfCameras * 16);
        cameras_pose.emplace_back(Pose{
            .p_wc = Vec3(radius * std::cos(theta) - radius + i * 1.0f, radius * std::sin(theta), 1.0f * std::sin(2 * theta)),
            .q_wc = Quat(Eigen::AngleAxis<float>(theta, Vec3::UnitX())),
        });
    }
    std::vector<LineSegment3D> line_segments_3d;
    for (int32_t i = 0; i < kNumberOfLines; ++i) {
        const float theta = i / 2.0f;
        line_segments_3d.emplace_back(LineSegment3D{
            Vec3(- 2.0f * i + std::cos(theta), std::sin(theta), 9 + i),
            Vec3(- i - std::cos(theta), - std::sin(theta), 7 + i + theta)
        });
    }

    // Generate vertex of cameras and lines.
    std::array<std::unique_ptr<Vertex<Scalar>>, kNumberOfCameras> all_camera_pos;
    std::array<std::unique_ptr<VertexQuat<Scalar>>, kNumberOfCameras> all_camera_rot;
    for (int32_t i = 0; i < kNumberOfCameras; ++i) {
        all_camera_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
        all_camera_pos[i]->param() = cameras_pose[i].p_wc.cast<Scalar>();
        all_camera_rot[i] = std::make_unique<VertexQuat<Scalar>>();
        all_camera_rot[i]->param() << cameras_pose[i].q_wc.w(), cameras_pose[i].q_wc.x(),
            cameras_pose[i].q_wc.y(), cameras_pose[i].q_wc.z();
        // TODO: fix bug of rotation jacobian, and remove it.
        all_camera_rot[i]->SetFixed(true);
    }
    std::array<std::unique_ptr<VertexLine<Scalar>>, kNumberOfLines> all_lines;
    for (int32_t i = 0; i < kNumberOfLines; ++i) {
        all_lines[i] = std::make_unique<VertexLine<Scalar>>();
        const LineSegment3D noised_line_3d = LineSegment3D(
            line_segments_3d[i].start_point() + Vec3::Random(),
            line_segments_3d[i].end_point() + Vec3::Random());
        all_lines[i]->param() = LinePlucker3D(LineSegment3D(noised_line_3d)).param().cast<Scalar>();
    }

    // Generate edge of observations.
    std::array<std::unique_ptr<EdgeOrthonormalLineToNormPlane<Scalar>>,
        kNumberOfCameras * kNumberOfLines> reprojection_edges;
    for (int32_t i = 0; i < kNumberOfCameras; ++i) {
        for (int32_t j = 0; j < kNumberOfLines; ++j) {
            const int32_t idx = i * kNumberOfLines + j;
            reprojection_edges[idx] = std::make_unique<EdgeOrthonormalLineToNormPlane<Scalar>>();
            reprojection_edges[idx]->SetVertex(all_lines[j].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[i].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[i].get(), 2);
            // Compute observation.
            const Vec3 p1_c = cameras_pose[i].q_wc.inverse() * (line_segments_3d[j].start_point() - cameras_pose[i].p_wc);
            const Vec3 p2_c = cameras_pose[i].q_wc.inverse() * (line_segments_3d[j].end_point() - cameras_pose[i].p_wc);
            reprojection_edges[idx]->observation() = LineSegment2D(p1_c.head<2>() / p1_c.z(), p2_c.head<2>() / p2_c.z()).param().cast<Scalar>();
            // Do self check.
            reprojection_edges[idx]->SelfCheck();
            reprojection_edges[idx]->SelfCheckJacobians();
        }
    }

    // Fix first two camera position.
    if (cameras_pose.size() > 2) {
        all_camera_pos[0]->SetFixed(true);
        all_camera_pos[1]->SetFixed(true);
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
    solver.options();
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
    // Draw observations.
    for (int32_t i = 0; i < kNumberOfCameras; ++i) {
        for (int32_t j = 0; j < kNumberOfLines; ++j) {
            const Vec3 p1_c = cameras_pose[i].q_wc.inverse() * (line_segments_3d[j].start_point() - cameras_pose[i].p_wc);
            const Vec3 p2_c = cameras_pose[i].q_wc.inverse() * (line_segments_3d[j].end_point() - cameras_pose[i].p_wc);
            Visualizor3D::lines().emplace_back(LineType{
                .p_w_i = cameras_pose[i].q_wc * p1_c / p1_c.z() + cameras_pose[i].p_wc,
                .p_w_j = cameras_pose[i].q_wc * p2_c / p2_c.z() + cameras_pose[i].p_wc,
                .color = RgbColor::kYellow,
            });
        }
    }
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
            .p_wc = Vec3(all_camera_pos[i]->param().cast<float>()),
            .q_wc = q_wc,
            .scale = 0.3f,
        });
    }
    for (uint32_t i = 0; i < all_lines.size(); ++i) {
        const auto &line = all_lines[i];
        const auto &line_segment = line_segments_3d[i];
        const LinePlucker3D plucker(Vec6(line->param().cast<float>()));
        Visualizor3D::lines().emplace_back(LineType{
            .p_w_i = plucker.ProjectPointOnLine(line_segment.start_point()),
            .p_w_j = plucker.ProjectPointOnLine(line_segment.end_point()),
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
