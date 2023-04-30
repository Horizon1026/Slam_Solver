#include "datatype_basic.h"
#include "log_api.h"

#include "vertex.h"
#include "vertex_quaternion.h"
#include "edge.h"
#include "math_kinematics.h"
#include "solver_lm.h"

using Scalar = float;
using namespace SLAM_SOLVER;

constexpr int32_t kCameraFrameNumber = 5;
constexpr int32_t kPointsNumber = 50;

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

        pixel_norm_xy = this->observation();

        p_c = q_wc.inverse() * (p_w - p_wc);
        inv_depth = static_cast<Scalar>(1) / p_c.z();

        if (std::isinf(inv_depth) || std::isnan(inv_depth)) {
            this->residual().setZero(2);
        } else {
            this->residual() = (p_c * inv_depth).template head<2>() - pixel_norm_xy;
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

    // Use string to represent edge type.
    virtual std::string GetType() { return std::string("Edge Reprojection"); }

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
    LogInfo(YELLOW ">> Test general graph optimizor on bundle adjustment." RESET_COLOR);

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
    }
    all_camera_pos.back()->param() += TVec3<Scalar>(1, 0, 0);

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
            reprojection_edges[idx] = std::make_unique<EdgeReproject<Scalar>>(2, 3);
            reprojection_edges[idx]->SetVertex(all_points[i].get(), 0);
            reprojection_edges[idx]->SetVertex(all_camera_pos[j].get(), 1);
            reprojection_edges[idx]->SetVertex(all_camera_rot[j].get(), 2);

            TVec3<Scalar> p_c = cameras[j].q_wc.inverse() * (points[i] - cameras[j].p_wc);
            TVec2<Scalar> obv = p_c.head<2>() / p_c.z();
            reprojection_edges[idx]->observation() = obv;
            reprojection_edges[idx]->SelfCheck();
        }
    }

    // Fix first camera pose.
    all_camera_pos.front()->SetFixed(true);
    all_camera_rot.front()->SetFixed(true);

    // Construct graph problem and solver.
    SolverLm<Scalar> solver;
    for (auto &vertex : all_camera_pos) { solver.problem().AddVertex(vertex.get()); }
    for (auto &vertex : all_camera_rot) { solver.problem().AddVertex(vertex.get()); }
    for (auto &vertex : all_points) { solver.problem().AddVertex(vertex.get(), false); }
    for (auto &edge : reprojection_edges) { solver.problem().AddEdge(edge.get()); }
    solver.Solve(false);

    // Show optimization result.
    for (int32_t i = 0; i < kPointsNumber; ++i) {
        LogInfo("[Point pos] [truth] " << LogVec(points[i]) << " | [result] " << LogVec(all_points[i]->param()));
    }
    for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
        LogInfo("[Camera pos] [truth] " << LogVec(cameras[i].p_wc) << " | [result] " << LogVec(all_camera_pos[i]->param()));
    }
    for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
        LogInfo("[Camera quat] [truth] " << LogQuat(cameras[i].q_wc) << " | [result] " << LogVec(all_camera_rot[i]->param()));
    }

    return 0;
}
