import numpy as np
import open3d as o3d

# 加載頭型3D模型和面部特徵點
head_mesh = o3d.io.read_triangle_mesh('../test_data/000220.obj')
facial_landmarks = np.load('../test_data/tracking.npy')

# 定義控制網格或骨架
control_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)

# 將頭型3D模型轉換到度量3D空間
transform = np.eye(4)
transform[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
head_mesh.transform(transform)

# 將面部特徵點映射到控制網格上
arr = np.array(head_mesh.vertices)
print(np.linalg.norm(arr[1]))
tree = o3d.geometry.KDTreeFlann(head_mesh.vertices)
_, idx, _ = tree.search_knn_vector_3d(head_mesh.vertices[0], 50)
print(idx)
# print(tree)
# control_points = []
# for landmark in facial_landmarks:

#     _, idx, _ = tree.search_knn_vector_3d(landmark, 1)
#     control_points.append(head_mesh.vertices[idx[0]])

# # 在控制網格上應用插值算法，將面部特徵點映射到頭型3D模型的頂點上
# interp_points = []
# for i in range(len(control_points)):
#     p1 = control_points[i % len(control_points)]
#     p2 = control_points[(i + 1) % len(control_points)]
#     interp_points.append((p1 + p2) / 2)
# interp_points = np.array(interp_points)

# # 將頭型3D模型的頂點轉換回原始空間
# transform_inv = np.linalg.inv(transform)
# head_mesh.transform(transform_inv)

# # 替換頭型3D模型的頂點位置
# head_mesh.vertices = o3d.utility.Vector3dVector(interp_points)

# # 保存新的頭型3D模型
# o3d.io.write_triangle_mesh('mapped_head_model.obj', head_mesh)
