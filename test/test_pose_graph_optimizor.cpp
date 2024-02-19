#include "datatype_basic.h"
#include "log_report.h"
#include "tick_tock.h"
#include "math_kinematics.h"
#include "visualizor_3d.h"

#include "pose_graph_optimizor.h"

using Scalar = float;
using namespace SLAM_SOLVER;
using namespace SLAM_VISUALIZOR;

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
        pose.p_wb = TVec3<Scalar>(0, 0, 0) - pose.q_wb * TVec3<Scalar>(0, 0, 8);
        poses.emplace_back(pose);
    }
}

void AddAllCamerasPoseAndPointsPosition(const std::vector<Pose<Scalar>> &poses) {
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

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test linear pose graph optimizor." RESET_COLOR);

    std::vector<Pose<Scalar>> poses;
    GenerateSimulationData(poses);

    // TODO: do pose graph optimization.

    // Add poses for visualizor.
    AddAllCamerasPoseAndPointsPosition(poses);

    do {
        Visualizor3D::Refresh("Visualizor", 50);
    } while (!Visualizor3D::ShouldQuit());

    return 0;
}
