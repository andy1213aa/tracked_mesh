import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform,
    TexturesUV,
    FoVPerspectiveCameras,
    PointLights,
    MeshRenderer,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams,
    Textures,
    PerspectiveCameras

)
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from PIL import Image
from pytorch3d.io import load_objs_as_meshes, load_obj

    
verts, faces, aux = load_obj('../test_data/000220.obj')

# verts = verts * 0.001

verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)

texture_image = Image.open("../test_data/000220.png")
texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
texture_image = np.array(texture_image)[::-1, :, :].copy()
texture_image = torch.from_numpy(texture_image).float() / 255.0
texture_image = texture_image.reshape((1, 1024, 1024, 3))

tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

meshes = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex.to(device))

# We scale normalize and center the target mesh to fit in a sphere of radius 1 
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
# to its original center and scale.  Note that normalizing the target mesh, 
# speeds up the optimization but is not necessary!
verts = meshes.verts_packed()
N = verts.shape[0]
center = verts.mean(0)
scale = max((verts - center).abs().max(0)[0])
meshes.offset_verts_(-center)
meshes.scale_verts_((1.0 / float(scale)))

# 設置攝像機和照明
R, T = look_at_view_transform(2.5, 0, 180)

focal = torch.tensor([5.0], dtype=torch.float32).to(device)
princpt = torch.tensor([0.1, 0.1], dtype=torch.float32).to(device).unsqueeze(0)

cameras = PerspectiveCameras(device=device, focal_length=focal, R=R, T=T, principal_point=princpt)
raster_settings = RasterizationSettings(
            image_size=[640, 480],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])

with torch.no_grad():
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(meshes)