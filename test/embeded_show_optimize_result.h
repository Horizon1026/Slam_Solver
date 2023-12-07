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
