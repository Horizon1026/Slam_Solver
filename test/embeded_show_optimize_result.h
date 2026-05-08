const int32_t max_points_num_to_print = std::min(10, kPointsNumber);
for (int32_t i = 0; i < max_points_num_to_print; ++i) {
    if (all_points[i]->param().rows() == points[i].rows()) {
        ReportInfo("[Point pos] [truth] " << LogVec(points[i]) << " | [result] " << LogVec(all_points[i]->param()));
    } else {
        const TVec3<Scalar> p_c0 = cameras[0].q_wc.inverse() * (points[i] - cameras[0].p_wc);
        const TVec3<Scalar> p_c_ = p_c0 / p_c0.z() / all_points[i]->param()(0);
        const TVec3<Scalar> p_w = cameras[0].q_wc * p_c_ + cameras[0].p_wc;
        ReportInfo("[Point pos] [truth] " << LogVec(points[i]) << " | [result] " << LogVec(p_w));
    }
}
for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
    ReportInfo("[Camera pos] [truth] " << LogVec(cameras[i].p_wc) << " | [result] " << LogVec(all_camera_pos[i]->param()));
}
for (int32_t i = 0; i < kCameraFrameNumber; ++i) {
    ReportInfo("[Camera quat] [truth] " << LogQuat(cameras[i].q_wc) << " | [result] " << LogVec(all_camera_rot[i]->param()));
}

// Visualize.
Visualizor3D::Clear();
// Draw ground truth of camera pose and points in world frame.
for (const auto &camera: cameras) {
    Visualizor3D::poses().emplace_back(PoseType {
        .p_wb = TVec3<float>(camera.p_wc.template cast<float>()),
        .q_wb = TQuat<float>(camera.q_wc.template cast<float>()),
        .scale = 0.5f,
    });
}
for (const auto &point: points) {
    Visualizor3D::points().emplace_back(PointType {
        .p_w = TVec3<float>(point.template cast<float>()),
        .color = RgbColor::kRed,
    });
}

// Draw result of optimization.
for (uint32_t i = 0; i < all_camera_pos.size(); ++i) {
    TQuat<Scalar> q_wc(all_camera_rot[i]->param()(0), all_camera_rot[i]->param()(1), all_camera_rot[i]->param()(2), all_camera_rot[i]->param()(3));
    Visualizor3D::camera_poses().emplace_back(CameraPoseType {
        .p_wc = TVec3<float>(all_camera_pos[i]->param().template cast<float>()),
        .q_wc = TQuat<float>(q_wc.template cast<float>()),
        .scale = 0.3f,
    });
}
for (uint32_t i = 0; i < all_points.size(); ++i) {
    if (all_points[i]->param().rows() == points[i].rows()) {
        Visualizor3D::points().emplace_back(PointType {
            .p_w = TVec3<float>(all_points[i]->param().template cast<float>()),
            .color = RgbColor::kCyan,
        });
    } else {
        const TVec3<Scalar> p_c0 = cameras[0].q_wc.inverse() * (points[i] - cameras[0].p_wc);
        const TVec3<Scalar> p_c_ = p_c0 / p_c0.z() / all_points[i]->param()(0);
        const TVec3<Scalar> p_w = cameras[0].q_wc * p_c_ + cameras[0].p_wc;
        Visualizor3D::points().emplace_back(PointType {
            .p_w = TVec3<float>(p_w.template cast<float>()),
            .color = RgbColor::kCyan,
        });
    }
}

Visualizor3D::camera_view().p_wc = TVec3<float>(0, 0, -10);
Visualizor3D::camera_view().q_wc = TQuat<float>::Identity();
while (!Visualizor3D::ShouldQuit()) {
    Visualizor3D::Refresh("Visualizor 3D", 30);
}
