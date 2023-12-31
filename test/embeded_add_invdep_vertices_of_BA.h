// Generate vertex of cameras and points.
std::array<std::unique_ptr<Vertex<Scalar>>, kCameraFrameNumber> all_camera_pos = {};
std::array<std::unique_ptr<VertexQuat<Scalar>>, kCameraFrameNumber> all_camera_rot = {};
for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
    all_camera_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
    all_camera_pos[i]->param() = cameras[i].p_wc;
    all_camera_pos[i]->name() = std::string("p_wc") + std::to_string(i);
    all_camera_rot[i] = std::make_unique<VertexQuat<Scalar>>(4, 3);
    all_camera_rot[i]->param() << cameras[i].q_wc.w(), cameras[i].q_wc.x(), cameras[i].q_wc.y(), cameras[i].q_wc.z();
    all_camera_rot[i]->name() = std::string("q_wc") + std::to_string(i);

    if (i > 2) {
        all_camera_pos[i]->param() += TVec3<Scalar>(0.5, 0.5, 0.5);
    }
}

std::array<std::unique_ptr<Vertex<Scalar>>, kPointsNumber> all_points = {};
for (int32_t i = 0; i < kPointsNumber; ++i) {
    const TVec3<Scalar> p_c = cameras[0].q_wc.inverse() * (points[i] - cameras[0].p_wc);
    const Scalar invdep = static_cast<Scalar>(1) / (p_c.z() + 1.0f);

    all_points[i] = std::make_unique<Vertex<Scalar>>(1, 1);
    all_points[i]->param() = TVec1<Scalar>(invdep);
}
