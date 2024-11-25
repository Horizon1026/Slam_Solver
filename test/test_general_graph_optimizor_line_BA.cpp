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

/* Class Vertex line. */
// TODO:

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
        // Get vertices.
        const LinePlucker3D line_plucker_w(Vec6(this->GetVertex(0)->param()));
        d_w = line_plucker_w.direction_vector().cast<Scalar>();
        n_w = line_plucker_w.normal_vector().cast<Scalar>();
        p_wc = this->GetVertex(1)->param();
        const TVec4<Scalar> &parameter = this->GetVertex(2)->param();
        q_wc = TQuat<Scalar>(parameter(0), parameter(1), parameter(2), parameter(3));
        // Get Observations.
        const LineSegment2D line_2d(this->observation());
        s_point = line_2d.start_point_homogeneous();
        e_point = line_2d.end_point_homogeneous();
        // Do reprojection.
        const LineOrthonormal3D line_orthonormal_w(line_plucker_w);
        matrix_U = line_orthonormal_w.matrix_U();
        matrix_W = line_orthonormal_w.matrix_W();
        const LinePlucker3D line_plucker_c = line_plucker_w.TransformTo(q_wc, p_wc);
        l = line_plucker_c.ProjectToNormalPlane();
        // Compute residual.
        this->residual() = Vec2(s_point.dot(l) / l.norm(), e_point.dot(l) / l.norm());
    }

    virtual void ComputeJacobians() override {
        const float l_squared_norm = l.head<2>().squaredNorm();
        const float l_norm = std::sqrt(l_squared_norm);
        const float l_squared_3_2 = l_squared_norm * l_norm;
        // Compute jacobian of d_residual to d_line_in_c.
        Mat2x3 jacobian_residual_line_in_c = Mat2x3::Zero();
        jacobian_residual_line_in_c << - l[0] * s_point.dot(l) / l_squared_3_2 + s_point.x() / l_norm,
                                        - l[1] * s_point.dot(l) / l_squared_3_2 + s_point.y() / l_norm,
                                        1.0f / l_norm,
                                        - l[0] * e_point.dot(l) / l_squared_3_2 + e_point.x() / l_norm,
                                        - l[1] * e_point.dot(l) / l_squared_3_2 + e_point.y() / l_norm,
                                        1.0f / l_norm;
        // Compute jacobian of d_line_in_c to d_plucker_in_c.
        Mat3x6 jacobian_line_to_plucker = Mat3x6::Zero();
        jacobian_line_to_plucker.block<3, 3>(0, 0).setIdentity();
        // Compute jacobian of d_plucker_in_c to d_plucker_in_w.
        Mat6 jacobian_plucker_c_to_w = Mat6::Zero();
        const Mat3 R_wc(q_wc);
        jacobian_plucker_c_to_w.block<3, 3>(0, 0) = R_wc;
        jacobian_plucker_c_to_w.block<3, 3>(3, 3) = R_wc;
        jacobian_plucker_c_to_w.block<3, 3>(0, 3) = Utility::SkewSymmetricMatrix(p_wc) * R_wc;
        // Compute jacobian of d_plucker_in_w to d_orthonormal_in_w.
        Mat6x4 jacobian_plucker_to_orthonormal = Mat6x4::Zero();
        const float w1 = matrix_W(0, 0);
        const float w2 = matrix_W(1, 0);
        jacobian_plucker_to_orthonormal.block<3, 1>(0, 1) = - w1 * matrix_U.col(2);
        jacobian_plucker_to_orthonormal.block<3, 1>(0, 2) = w1 * matrix_U.col(1);
        jacobian_plucker_to_orthonormal.block<3, 1>(0, 3) = - w2 * matrix_U.col(0);
        jacobian_plucker_to_orthonormal.block<3, 1>(3, 0) = w2 * matrix_U.col(2);
        jacobian_plucker_to_orthonormal.block<3, 1>(3, 2) = - w2 * matrix_U.col(0);
        jacobian_plucker_to_orthonormal.block<3, 1>(3, 3) = w1 * matrix_U.col(1);
        // Compute jacobian of d_plucker_in_c to d_cam_p_wc.
        Mat6x3 jacobian_plucker_c_to_p_wc = Mat6x3::Zero();
        jacobian_plucker_c_to_p_wc.block<3, 3>(0, 0) = q_wc.matrix().transpose() * Utility::SkewSymmetricMatrix(d_w);
        // Compute jacobian of d_plucker_in_c to d_cam_q_wc.
        Mat6x3 jacobian_plucker_c_to_q_wc = Mat6x3::Zero();
        const Vec3 temp_vec = q_wc.inverse() * (n_w + Utility::SkewSymmetricMatrix(d_w) * p_wc);
        jacobian_plucker_c_to_q_wc.block<3, 3>(0, 0) = Utility::SkewSymmetricMatrix(temp_vec);
        jacobian_plucker_c_to_q_wc.block<3, 3>(3, 0) = Utility::SkewSymmetricMatrix(Vec3(q_wc.matrix().transpose() * d_w));
        // Compute full jacobian.
        const auto jacobian_residual_to_plucker_in_c = jacobian_residual_line_in_c * jacobian_line_to_plucker;
        this->GetJacobian(0) = jacobian_residual_to_plucker_in_c *
                               jacobian_plucker_c_to_w *
                               jacobian_plucker_to_orthonormal;
        this->GetJacobian(1) = jacobian_residual_to_plucker_in_c *
                               jacobian_plucker_c_to_p_wc;
        this->GetJacobian(2) = jacobian_residual_to_plucker_in_c *
                               jacobian_plucker_c_to_q_wc;
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    Vec3 d_w = Vec3::Zero();
    Vec3 n_w = Vec3::Zero();
    Vec3 p_wc = Vec3::Zero();
    Quat q_wc = Quat::Identity();
    Mat3 matrix_U = Mat3::Zero();
    Mat2 matrix_W = Mat2::Zero();
    Vec3 s_point = Vec3::Zero();
    Vec3 e_point = Vec3::Zero();
    Vec3 l = Vec3::Zero();

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
        all_lines[i] = std::make_unique<Vertex<Scalar>>(6, 4);
        all_lines[i]->param() = LinePlucker3D(LineSegment3D(line_segments_3d[i])).param().cast<Scalar>();
    }

    // TODO:
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
