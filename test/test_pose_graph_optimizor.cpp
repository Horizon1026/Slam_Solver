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
    TQuat<Scalar> q_wc = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wc = TVec3<Scalar>::Zero();
};

void GenerateSimulationData(std::vector<Pose<Scalar>> &cameras,
                            std::vector<TVec3<Scalar>> &points) {
    cameras.clear();
    points.clear();

    // Cameras.
    for (int32_t i = 0; i < 20; ++i) {
        Pose<Scalar> camera_pose;
        const TVec3<Scalar> euler = TVec3<Scalar>(0, -90 + i * 18, 0);
        camera_pose.q_wc = Utility::EulerToQuaternion(euler);
        camera_pose.p_wc = TVec3<Scalar>(0, 0, 0) - camera_pose.q_wc * TVec3<Scalar>(0, 0, 8);
        cameras.emplace_back(camera_pose);
    }

    // Points.
    int32_t offset = -2;
    for (int32_t i = 0; i < 5; ++i) {
        for (int32_t j = 0; j < 5; ++j) {
            for (int32_t k = 0; k < 5; ++k) {
                const TVec3<Scalar> point(i + offset, j + offset, k + offset);
                points.emplace_back(point);
            }
        }
    }
}

void AddAllCamerasPoseAndPointsPosition(const std::vector<Pose<Scalar>> &cameras,
                                        const std::vector<TVec3<Scalar>> &points) {
    Visualizor3D::Clear();

    // Add word frame.
    Visualizor3D::poses().emplace_back(PoseType{
        .p_wb = Vec3::Zero(),
        .q_wb = Quat::Identity(),
        .scale = 10.0f,
    });

    // Add all points.
    for (const auto &point : points) {
        Visualizor3D::points().emplace_back(PointType{
            .p_w = point.cast<float>(),
            .color = RgbColor::kCyan,
            .radius = 2,
        });
    }

    // Add all cameras pose.
    for (uint32_t i = 0; i < cameras.size(); ++i) {
        Visualizor3D::poses().emplace_back(PoseType{
            .p_wb = cameras[i].p_wc,
            .q_wb = cameras[i].q_wc,
            .scale = 1.0,
        });

        if (i) {
            Visualizor3D::lines().emplace_back(LineType{
                .p_w_i = cameras[i - 1].p_wc,
                .p_w_j = cameras[i].p_wc,
                .color = RgbColor::kWhite,
            });
        }
    }
}

int main(int argc, char **argv) {
    LogFixPercision(3);
    ReportInfo(YELLOW ">> Test linear pose graph optimizor." RESET_COLOR);

    std::vector<Pose<Scalar>> cameras;
    std::vector<TVec3<Scalar>> points;
    GenerateSimulationData(cameras, points);

    // TODO: do pose graph optimization.

    // Add cameras pose and points position.
    AddAllCamerasPoseAndPointsPosition(cameras, points);

    do {
        Visualizor3D::Refresh("Visualizor", 50);
    } while (!Visualizor3D::ShouldQuit());

    return 0;
}
