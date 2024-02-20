#include "datatype_basic.h"
#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"
#include "visualizor_3d.h"

#include "general_graph_optimizor.h"
#include "pose_graph_optimizor.h"

#include "enable_stack_backward.h"

using Scalar = float;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;

/* Simulation Data. */
template <typename Scalar>
struct Pose {
    TQuat<Scalar> q_wb = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wb = TVec3<Scalar>::Zero();
};

void GenerateSimulationData(std::vector<Pose<Scalar>> &poses) {
    poses.clear();

    // Poses.
    for (int32_t i = 0; i < 20; ++i) {
        Pose<Scalar> pose;
        const TVec3<Scalar> euler = TVec3<Scalar>(0, -90 + i * 18, 0);
        pose.q_wb = Utility::EulerToQuaternion(euler);
        pose.p_wb = TVec3<Scalar>(0, 0, 0) - pose.q_wb * TVec3<Scalar>(0, 0, 8 + i * 0.1);
        poses.emplace_back(pose);
    }
}

/* Class Edge Relative Pose. */
template <typename Scalar>
class EdgeRelativePose : public Edge<Scalar> {
// Vertices : [ref_pose, p_wb0]
//            [ref_pose, q_wb0]
//            [cur_pose, p_wb1]
//            [cur_pose, q_wb1]

public:
    EdgeRelativePose() : Edge<Scalar>(6, 4) {}
    virtual ~EdgeRelativePose() = default;

    // Compute residual and jacobians for each vertex. These operations should be defined by subclass.
    virtual void ComputeResidual() override {

    }

    virtual void ComputeJacobians() override {

    }

private:
    // Parameters will be calculated in ComputeResidual().
    // It should not be repeatedly calculated in ComputeJacobians().
    TVec3<Scalar> p_b0b1 = TVec3<Scalar>::Zero();
    TQuat<Scalar> q_b0b1 = TQuat<Scalar>::Identity();
};

void AddAllRawPosesIntoVisualizor(const std::vector<Pose<Scalar>> &poses) {
    Visualizor3D::Clear();

    // Add word frame.
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 10.0f,
    });
    Visualizor3D::points().emplace_back(PointType{
        .p_w = Vec3::Zero(),
        .color = RgbColor::kWhite,
        .radius = 2,
    });

    // Add all poses.
    for (uint32_t i = 0; i < poses.size(); ++i) {
        Visualizor3D::poses().emplace_back(PoseType{
            .p_wb = poses[i].p_wb,
            .q_wb = poses[i].q_wb,
            .scale = 1.0,
        });

        if (i) {
            Visualizor3D::lines().emplace_back(LineType{
                .p_w_i = poses[i - 1].p_wb,
                .p_w_j = poses[i].p_wb,
                .color = RgbColor::kWhite,
            });
        }
    }
}

void DoPgoByGeneralGraphOptimizor(const std::vector<Pose<Scalar>> &poses) {
    // TODO:
}

void DoPgoByPoseGraphOptimizor(const std::vector<Pose<Scalar>> &poses) {
    // TODO:
}

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test linear pose graph optimizor." RESET_COLOR);

    std::vector<Pose<Scalar>> poses;
    GenerateSimulationData(poses);
    // Add raw poses for visualizor.
    AddAllRawPosesIntoVisualizor(poses);

    // Do pose graph optimization.
    DoPgoByGeneralGraphOptimizor(poses);
    DoPgoByPoseGraphOptimizor(poses);

    do {
        Visualizor3D::Refresh("Visualizor", 50);
    } while (!Visualizor3D::ShouldQuit());

    return 0;
}
