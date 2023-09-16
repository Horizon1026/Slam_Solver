#include "datatype_basic.h"
#include "log_report.h"
#include "tick_tock.h"

#include "vertex.h"
#include "vertex_quaternion.h"
#include "edge.h"
#include "math_kinematics.h"
#include "solver_lm.h"
#include "solver_dogleg.h"

using Scalar = float;
using namespace SLAM_SOLVER;

constexpr int32_t kCameraFrameNumber = 10;
constexpr int32_t kPointsNumber = 300;

/* Class Edge reprojection. */
template <typename Scalar>
class EdgeReproject : public Edge<Scalar> {
// vertex is [feature, p_w] [camera, p_wc] [camera, q_wc].

public:
    EdgeReproject() = delete;
    EdgeReproject(int32_t residual_dim, int32_t vertex_num) : Edge<Scalar>(residual_dim, vertex_num) {}
    virtual ~EdgeReproject() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {
        p_w = this->GetVertex(0)->param();
        p_wc = this->GetVertex(1)->param();
        const TVec4<Scalar> &parameter = this->GetVertex(2)->param();
        q_wc = TQuat<Scalar>(parameter(0), parameter(1), parameter(2), parameter(3));
        TVec3<Scalar> sphere_xyz = this->observation();
        p_c = q_wc.inverse() * (p_w - p_wc);
        inv_depth = static_cast<Scalar>(1) / p_c.z();

        const TVec2<Scalar> pred_norm_xy = (p_c * inv_depth).template head<2>();
        yita = 2.0 / (1.0 + pred_norm_xy.squaredNorm());
        const TVec3<Scalar> pred_sphere_xyz = TVec3<Scalar>(pred_norm_xy.x() * yita, pred_norm_xy.y() * yita, yita - 1.0);

        if (std::isinf(inv_depth) || std::isnan(inv_depth)) {
            this->residual().setZero(3);
        } else {
            this->residual() = pred_sphere_xyz - sphere_xyz;
        }
    }

    virtual void ComputeJacobians() override {
        TMat3<Scalar> jacobian_sphere_3d = TMat3<Scalar>::Zero();
        if (!std::isinf(inv_depth) && !std::isnan(inv_depth)) {
            TMat3x2<Scalar> jacobian_sphere_norm = TMat3x2<Scalar>::Zero();
            const Scalar yita2_4 = yita * yita * 4.0;
            const Scalar uv = pred_norm_xy(0) * pred_norm_xy(1);
            const Scalar u2 = pred_norm_xy(0) * pred_norm_xy(0);
            const Scalar v2 = pred_norm_xy(1) * pred_norm_xy(1);
            jacobian_sphere_norm << yita - u2 * yita2_4, yita - v2 * yita2_4,
                                    - uv * yita2_4, - uv * yita2_4,
                                    - pred_norm_xy(0) * yita2_4, - pred_norm_xy(1) * yita2_4;

            TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
            const Scalar inv_depth_2 = inv_depth * inv_depth;
            jacobian_2d_3d << inv_depth, 0, - p_c(0) * inv_depth_2,
                              0, inv_depth, - p_c(1) * inv_depth_2;

            jacobian_sphere_3d = jacobian_sphere_norm * jacobian_2d_3d;
        }

        this->GetJacobian(0) = jacobian_sphere_3d * (q_wc.inverse().matrix());
        this->GetJacobian(1) = - this->GetJacobian(0);
        this->GetJacobian(2) = jacobian_sphere_3d * SLAM_UTILITY::Utility::SkewSymmetricMatrix(p_c);
    }

    // Use string to represent edge type.
    virtual std::string GetType() { return std::string("Edge Reprojection"); }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_w;
    TVec3<Scalar> p_wc;
    TQuat<Scalar> q_wc;
    TVec3<Scalar> p_c;
    Scalar inv_depth = 0;
    Scalar yita = 0;
    TVec2<Scalar> pred_norm_xy;
};

template <typename Scalar>
struct Pose {
    TQuat<Scalar> q_wc = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wc = TVec3<Scalar>::Zero();
};

void GenerateSimulationData(std::vector<Pose<Scalar>> &cameras,
                            std::vector<TVec3<Scalar>> &points) {
    cameras.clear();
    points.clear();

    // Cameras.
    for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
        Pose<Scalar> camera_pose;
        camera_pose.q_wc.setIdentity();
        camera_pose.p_wc = TVec3<Scalar>(i, i, 0);
        cameras.emplace_back(camera_pose);
    }

    // Points.
    for (int32_t i = 0; i < 20; ++i) {
        for (int32_t j = 0; j < 20; ++j) {
            const TVec3<Scalar> point(i, j, 5);
            points.emplace_back(point);
            if (points.size() == kPointsNumber) {
                return;
            }
        }
    }
}

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with norm plane model." RESET_COLOR);

    std::vector<Pose<Scalar>> cameras;
    std::vector<TVec3<Scalar>> points;
    GenerateSimulationData(cameras, points);

    // Generate vertex of cameras and points.
    std::array<std::unique_ptr<Vertex<Scalar>>, kCameraFrameNumber> all_camera_pos = {};
    std::array<std::unique_ptr<VertexQuat<Scalar>>, kCameraFrameNumber> all_camera_rot = {};
    for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
        all_camera_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
        all_camera_pos[i]->param() = cameras[i].p_wc;
        all_camera_rot[i] = std::make_unique<VertexQuat<Scalar>>(4, 3);
        all_camera_rot[i]->param() << cameras[i].q_wc.w(), cameras[i].q_wc.x(), cameras[i].q_wc.y(), cameras[i].q_wc.z();

        if (i > 2) {
            all_camera_pos[i]->param() += TVec3<Scalar>(0.2, 0.2, 0);
        }
    }

    std::array<std::unique_ptr<Vertex<Scalar>>, kPointsNumber> all_points = {};
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        all_points[i] = std::make_unique<Vertex<Scalar>>(3, 3);
        all_points[i]->param() = points[i];
    }

    // Generate edges between cameras and points.
    std::array<std::unique_ptr<EdgeReproject<Scalar>>, kCameraFrameNumber * kPointsNumber> reprojection_edges = {};
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        for (int32_t j = 0; j < kCameraFrameNumber; ++j) {
            const int32_t idx = i * kCameraFrameNumber + j;
            reprojection_edges[idx] = std::make_unique<EdgeReproject<Scalar>>(3, 3);
            reprojection_edges[idx]->SetVertex(all_points[i].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[j].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[j].get(), 2);

            TVec3<Scalar> p_c = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec2<Scalar> norm_xy = p_c.head<2>() / p_c.z();
            const Scalar yita = 2.0 / (1.0 + norm_xy.squaredNorm());
            TVec3<Scalar> sphere_xyz = TVec3<Scalar>(norm_xy.x() * yita, norm_xy.y() * yita, yita - 1.0);

            reprojection_edges[idx]->observation() = sphere_xyz;
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
    const int32_t max_points_num_to_print = std::min(10, kPointsNumber);
    for (int32_t i = 0; i < max_points_num_to_print; ++i) {
        ReportInfo("[Point pos] [truth] " << LogVec(points[i]) << " | [result] " << LogVec(all_points[i]->param()));
    }
    for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
        ReportInfo("[Camera pos] [truth] " << LogVec(cameras[i].p_wc) << " | [result] " << LogVec(all_camera_pos[i]->param()));
    }
    for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
        ReportInfo("[Camera quat] [truth] " << LogQuat(cameras[i].q_wc) << " | [result] " << LogVec(all_camera_rot[i]->param()));
    }

    return 0;
}
