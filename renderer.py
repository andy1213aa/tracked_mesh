import torch
import numpy as np
from pytorch3d.renderer import (PointLights, MeshRenderer,
                                RasterizationSettings, MeshRasterizer,
                                SoftPhongShader, Textures, PerspectiveCameras)

from pytorch3d.structures import Meshes


class Renderer():

    def __init__(self, device: str = 'cuda:0'):
        self.device = device

    def _linear_transform(self, verts, transform):
        """
            對一個 batch 的頂點進行線性轉換
            Args:
                verts: shape (batch_size, num_verts, 3) 的頂點坐標張量
                transform: shape (batch_size, 3, 4) 的線性轉換矩陣張量
            Returns:
                shape (batch_size, num_verts, 3) 的轉換後的頂點坐標張量
        """
        batch_size, num_verts, _ = verts.shape
        R = transform[:, :3, :3]
        t = transform[:, :, 3].unsqueeze(2)
        verts = torch.bmm(R, verts.transpose(1, 2)).transpose(1, 2) + t
        return verts

    def render(self, verts, faces_uvs, verts_uvs, verts_idx, texture_image, transform_head,
               transform_camera, focal, princpt, render_width, render_height):

        verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
        faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)

        tex = Textures(verts_uvs=verts_uvs,
                       faces_uvs=faces_uvs,
                       maps=texture_image)

        verts_head = self._linear_transform(verts, transform_head)
        verts_cam = self._linear_transform(verts_head, transform_camera)

        meshes = Meshes(verts=[verts_cam.to(self.device)],
                        faces=[verts_idx.to(self.device)],
                        textures=tex.to(self.device))

        cameras = PerspectiveCameras(device=self.device,
                                     focal_length=-focal,
                                     principal_point=princpt,
                                     in_ndc=False,
                                     image_size=((render_height,
                                                  render_width), ))

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
            images *= 255
            return images