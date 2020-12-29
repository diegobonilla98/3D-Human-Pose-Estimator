import cv2
from MiDaS import depth_utils
import numpy as np
import open3d
import mediapipe as mp
import matplotlib.pyplot as plt


def create_3d_from_depth(le_map, frame):
    color_raw = frame[:, :, ::-1].astype('float32') / 255.
    colors = np.reshape(color_raw, (color_raw.shape[0] * color_raw.shape[1], 3))

    x = np.arange(0, le_map.shape[1])
    y = np.arange(0, le_map.shape[0])
    mesh_x, mesh_y = np.meshgrid(x, y)
    z = (le_map.astype('float32')) * 10.

    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z, -1)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

depth_utils.load_model()

image = cv2.imread('full-body-young-woman-white-background-person-pointing-by-hand-shirt-copy-space-proud-confident_1187-33135.jpg')

image_rows, image_cols = image.shape[:2]

out = depth_utils.predict_depth(image)
results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

landmarks = []

out = -75.629 + 254.129 * np.exp(-0.0011 * (out.astype('float32')))

landmark_list = results.pose_landmarks
idx_to_coordinates = {}
for idx, landmark in enumerate(landmark_list.landmark):
    if landmark.visibility < 0 or landmark.presence < 0:
        continue
    landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
        idx_to_coordinates[idx] = landmark_px
idx_to_coordinates = list(idx_to_coordinates.values())
width = max(idx_to_coordinates, key=lambda x: x[0])[0] - min(idx_to_coordinates, key=lambda x: x[0])[0]
height = (max(idx_to_coordinates, key=lambda x: x[1])[1] - min(idx_to_coordinates, key=lambda x: x[1])[1]) * 2
center = ((max(idx_to_coordinates, key=lambda x: x[0])[0] + min(idx_to_coordinates, key=lambda x: x[0])[0]) / 2,
          (max(idx_to_coordinates, key=lambda x: x[1])[1] + min(idx_to_coordinates, key=lambda x: x[1])[1]) / 2)
zs = []
for landmark_px in idx_to_coordinates:
    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=10, resolution=15)
    sphere.compute_vertex_normals()
    sphere.translate((landmark_px[0], landmark_px[1], 10. * out[landmark_px[1], landmark_px[0]]))
    sphere.paint_uniform_color([1, 0.706, 0])
    landmarks.append(sphere)
    zs.append(10. * out[landmark_px[1], landmark_px[0]])

pcd = create_3d_from_depth(out, image)

depth = max(zs) - min(zs)
box = open3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
box.translate((center[0] - width / 2, center[1] - height / 3, np.median(zs) - depth / 4))
line_box = open3d.geometry.LineSet.create_from_triangle_mesh(box)

open3d.visualization.draw_geometries(landmarks + [pcd, line_box])
