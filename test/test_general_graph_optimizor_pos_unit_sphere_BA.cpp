#include "datatype_basic.h"
#include "log_report.h"
#include "tick_tock.h"

#include "vertex.h"
#include "vertex_quaternion.h"
#include "edge.h"
#include "math_kinematics.h"
#include "solver_lm.h"
#include "solver_dogleg.h"

#include "enable_stack_backward.h"

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
        // Compute prediction.
        p_w = this->GetVertex(0)->param();
        p_wc = this->GetVertex(1)->param();
        const TVec4<Scalar> &parameter = this->GetVertex(2)->param();
        q_wc = TQuat<Scalar>(parameter(0), parameter(1), parameter(2), parameter(3));
        p_c = q_wc.inverse() * (p_w - p_wc);

        // Get observation.
        obv_norm_xy = this->observation();
        const TVec3<Scalar> obv_p_c = TVec3<Scalar>(obv_norm_xy.x(), obv_norm_xy.y(), 1.0f);

        // Compute residual.
        this->residual() = tangent_base_transpose * (p_c.normalized() - obv_p_c.normalized());
    }

    virtual void ComputeJacobians() override {
        const Scalar p_c_norm = p_c.norm();
        const Scalar p_c_norm3 = p_c_norm * p_c_norm * p_c_norm;
        TMat3<Scalar> jacobian_norm = TMat3<Scalar>::Zero();
        jacobian_norm << 1.0 / p_c_norm - p_c.x() * p_c.x() / p_c_norm3, - p_c.x() * p_c.y() / p_c_norm3,                - p_c.x() * p_c.z() / p_c_norm3,
                         - p_c.x() * p_c.y() / p_c_norm3,                1.0 / p_c_norm - p_c.y() * p_c.y() / p_c_norm3, - p_c.y() * p_c.z() / p_c_norm3,
                         - p_c.x() * p_c.z() / p_c_norm3,                - p_c.y() * p_c.z() / p_c_norm3,                1.0 / p_c_norm - p_c.z() * p_c.z() / p_c_norm3;

        TMat2x3<Scalar> jacobian_2d_3d = TMat2x3<Scalar>::Zero();
        jacobian_2d_3d = tangent_base_transpose * jacobian_norm;

        this->GetJacobian(0) = jacobian_2d_3d * (q_wc.inverse().matrix());
        this->GetJacobian(1) = - this->GetJacobian(0);
        this->GetJacobian(2) = jacobian_2d_3d * SLAM_UTILITY::Utility::SkewSymmetricMatrix(p_c);
    }

    // Use string to represent edge type.
    virtual std::string GetType() { return std::string("Edge Reprojection"); }

    // Set tangent base.
    void SetTrangetBase(const TVec3<Scalar> &vec) {
        tangent_base_transpose = Utility::TangentBase(vec).transpose();
    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_w;
    TVec3<Scalar> p_wc;
    TQuat<Scalar> q_wc;
    TVec2<Scalar> obv_norm_xy;
    TVec3<Scalar> p_c;
    TMat2x3<Scalar> tangent_base_transpose;
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
    ReportInfo(YELLOW ">> Test general graph optimizor on bundle adjustment with <pos> <unit sphere> model." RESET_COLOR);

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
            reprojection_edges[idx] = std::make_unique<EdgeReproject<Scalar>>(2, 3);
            reprojection_edges[idx]->SetVertex(all_points[i].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[j].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[j].get(), 2);

            TVec3<Scalar> p_c = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec2<Scalar> obv = p_c.head<2>() / p_c.z();
            reprojection_edges[idx]->SetTrangetBase(p_c);
            reprojection_edges[idx]->observation() = obv;
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
