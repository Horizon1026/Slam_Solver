constexpr int32_t kCameraFrameNumber = 10;
constexpr int32_t kPointsNumber = 300;
constexpr int32_t kCameraExtrinsicNumber = 1;

template <typename Scalar>
struct Pose {
    TQuat<Scalar> q_wc = TQuat<Scalar>::Identity();
    TVec3<Scalar> p_wc = TVec3<Scalar>::Zero();
};

void GenerateSimulationData(std::vector<Pose<Scalar>> &cameras, std::vector<TVec3<Scalar>> &points) {
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