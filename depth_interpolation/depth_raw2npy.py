import sys
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import json

# import argparse
assert len(sys.argv) > 1, 'need bag file path'
bagfile = sys.argv[1]

pipeline = rs.pipeline()
# Create a config并配置要流​​式传输的管道
config = rs.config()
config.enable_device_from_file(bagfile)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)

# 按照日期创建文件夹
# save_path = os.path.join(os.path.dirname(bagfile), "aligned_depth_color", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
save_path = os.path.join(os.path.dirname(bagfile), "aligned_depth_color")
bagname = os.path.basename(bagfile).split('.')[0]
# save_color_path = os.path.join(save_path, "color")
# save_depth_path = os.path.join(save_path, "depth")
# save_intrin_path = os.path.join(save_path, "intrin")
if not os.path.isdir(save_path):
    os.makedirs(save_path)

# if not os.path.isdir(save_color_path):
#     os.mkdir(save_color_path)

# if not os.path.isdir(save_depth_path):
#     os.mkdir(save_depth_path)

# if not os.path.isdir(save_intrin_path):
#     os.mkdir(save_intrin_path)

# 保存的图片和实时的图片界面
cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
saved_color_image = None  # 保存的临时图片
saved_depth_mapped_image = None
saved_count = 0

# 主循环
try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_data = np.asanyarray(
            aligned_depth_frame.get_data(), dtype="float16")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = color_image[:, :, ::-1]
        depth_mapped_image = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))
        key = cv2.waitKey(30)

        
        # depth_pixel = (50,50)
        # depth_value = depth_image[depth_pixel[0], depth_pixel[1]] * 0.001
        # world_coors = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)

        # s 保存图片
        if key & 0xFF == ord('s'):
            saved_color_image = color_image
            saved_depth_mapped_image = depth_mapped_image

            # 彩色图片保存为png格式
            cv2.imwrite(os.path.join(save_path, "{}-{}-color.png".format(bagname, saved_count)), saved_color_image)
            # 深度信息由采集到的float16直接保存为npy格式
            np.save(os.path.join(save_path, "{}-{}-depth".format(bagname, saved_count)), depth_data)
            # 相机内参保存未pickle格式
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            json_name = os.path.join(save_path, "{}-{}-intrin.json".format(bagname, saved_count))
            intrin_dict = {
                'depth_scale':depth_scale,
                'height':depth_intrin.height,
                'width':depth_intrin.width,
                'coeffs':depth_intrin.coeffs, 
                'fx':depth_intrin.fx, 
                'fy':depth_intrin.fy,
                'ppx':depth_intrin.ppx,
                'ppy':depth_intrin.ppy,
                'model':depth_intrin.model.__str__(),
            }
            
            # intri = rs.intrinsics()
            # intri.height = intrin_dict['height']
            # intri.width = intrin_dict['width']
            # intri.coeffs = intrin_dict['coeffs']
            # intri.fx = intrin_dict['fx']
            # intri.fy = intrin_dict['fy']
            # intri.ppx = intrin_dict['ppx']
            # intri.ppy = intrin_dict['ppy']
            # # rs_dis = rs.distortion(2)
            # intri.model = rs.distortion.inverse_brown_conrady
            # print('intri', intri)

            # depth_pixel = (50,50)
            # depth_value = depth_image[depth_pixel[0], depth_pixel[1]] * depth_scale
            # world_coors_1 = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
            # world_coors_2 = rs.rs2_deproject_pixel_to_point(intri, depth_pixel, depth_value)


            with open(json_name, 'w') as f:
                json.dump(intrin_dict, f)

            saved_count += 1
            cv2.imshow("save", np.hstack(
                (saved_color_image, saved_depth_mapped_image)))

        # q 退出
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
