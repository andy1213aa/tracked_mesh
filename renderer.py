import torch
import numpy as np
from pytorch3d.renderer import (look_at_view_transform, TexturesUV,
                                FoVPerspectiveCameras, PointLights,
                                MeshRenderer, RasterizationSettings,
                                MeshRasterizer, SoftPhongShader, BlendParams,
                                Textures, PerspectiveCameras)
import cv2
import matplotlib.pyplot as plt
from pytorch3d.structures import Meshes
from PIL import Image
from pytorch3d.io import load_objs_as_meshes, load_obj


class Renderer():

    def __init__(self, device: str = 'cuda:0'):
        self.device = device

    def render(self, verts, faces, aux, texture_image,
               transform_head, transform_camera, focal, princpt,
               render_width, render_height):
        
        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
        

        texture_image = Image.open("../test_data/000220.png")
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        texture_image = np.array(texture_image)[::-1, :, :].copy()
        texture_image = torch.from_numpy(texture_image).float() / 255.0
        texture_image = texture_image.reshape((1, 1024, 1024, 3))
        
        tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

        transform = np.array(transform_head).reshape((3, 4))
        R = transform[:3, :3]
        t = transform[:, 3]
        verts = torch.tensor(((R @ verts.numpy().T).T + t), dtype=torch.float32)

        transform = np.array(transform_camera).reshape((3, 4))
        R = transform[:3, :3]
        t = transform[:, 3]
        verts = torch.tensor(((R @ verts.numpy().T).T + t), dtype=torch.float32)

        meshes = Meshes(verts=[verts.to(self.device)],
                        faces=[faces.verts_idx.to(self.device)],
                        textures=tex.to(self.device))

        focal = torch.tensor([7702.4736, 7703.745],
                            dtype=torch.float32).to(self.device).unsqueeze(0)
        princpt = torch.tensor([801.6277, 997.59296],
                            dtype=torch.float32).to(self.device).unsqueeze(0)

        cameras = PerspectiveCameras(device=self.device,
                                    focal_length=-focal,
                                    principal_point=princpt,
                                    in_ndc=False,
                                    image_size=((render_height, render_width), ))
        # cameras = PerspectiveCameras(device=device, focal_length=focal, principal_point=princpt)
        raster_settings = RasterizationSettings(
            image_size=[render_height, render_width],
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=[[0.0, 1.0, -10.0]])

        with torch.no_grad():
            renderer = MeshRenderer(rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings),
                                    shader=SoftPhongShader(device=self.device,
                                                        cameras=cameras,
                                                        lights=lights))
            images = renderer(meshes, znear=0.0, zfar=1500.0)
