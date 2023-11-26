// Generate vertex of cameras and points.
std::array<std::unique_ptr<Vertex<Scalar>>, kCameraFrameNumber> all_camera_pos = {};
std::array<std::unique_ptr<VertexQuat<Scalar>>, kCameraFrameNumber> all_camera_rot = {};
for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
    all_camera_pos[i] = std::make_unique<Vertex<Scalar>>(3, 3);
    all_camera_pos[i]->param() = cameras[i].p_wc;
    all_camera_rot[i] = std::make_unique<VertexQuat<Scalar>>(4, 3);
    all_camera_rot[i]->param() << cameras[i].q_wc.w(), cameras[i].q_wc.x(), cameras[i].q_wc.y(), cameras[i].q_wc.z();

    if (i > 2) {
        all_camera_pos[i]->param() += TVec3<Scalar>(0.5, 0.5, 0.5);
    }
}

std::array<std::unique_ptr<Vertex<Scalar>>, kPointsNumber> all_points = {};
for (int32_t i = 0; i < kPointsNumber; ++i) {
    all_points[i] = std::make_unique<Vertex<Scalar>>(3, 3);
    all_points[i]->param() = points[i];
    all_points[i]->param() += TVec3<Scalar>(0.1, 0.1, 0.1);
}
