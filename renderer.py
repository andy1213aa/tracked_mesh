import torch
import numpy as np
import jax.dlpack as jdl

from pytorch3d.renderer import (PointLights, MeshRenderer,
                                RasterizationSettings, MeshRasterizer,
                                SoftPhongShader, Textures, PerspectiveCameras)

from pytorch3d.structures import Meshes


class Renderer():

    def __init__(self, image_size, render_size, device: str = 'cuda:0'):
        self.device = device
        self.image_size = image_size
        self.render_size = render_size

    def _split_batch_to_list(self, tensor):
        # split the tensor along the first dimension
        tensor = torch.split(tensor, split_size_or_sections=1, dim=0)
        # # remove the first dimension from each tensor in the list
        tensor = [t.squeeze(0).to(self.device) for t in tensor]
        return tensor

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
        t = transform[:, :, 3].unsqueeze(1)
        verts = torch.bmm(R, verts.transpose(1, 2)).transpose(1, 2) + t
        return verts

    def _jax2torch(self, verts, faces_uvs, verts_uvs, verts_idx, texture_image,
                   transform_head, transform_camera, focal, princpt):
        print(verts)
        return (torch.from_dlpack(jdl.to_dlpack(verts)),
                torch.from_dlpack(jdl.to_dlpack(faces_uvs)).to(dtype=torch.int64),
                torch.from_dlpack(jdl.to_dlpack(verts_uvs)),
                torch.from_dlpack(jdl.to_dlpack(verts_idx)).to(dtype=torch.int64),
                torch.from_dlpack(jdl.to_dlpack(texture_image)) / 255.0,
                torch.from_dlpack(jdl.to_dlpack(transform_head)),
                torch.from_dlpack(jdl.to_dlpack(transform_camera)),
                torch.from_dlpack(jdl.to_dlpack(focal)),
                torch.from_dlpack(jdl.to_dlpack(princpt)))

    def render(self,
               verts,
               faces_uvs,
               verts_uvs,
               verts_idx,
               texture_image,
               transform_head,
               transform_camera,
               focal,
               princpt,
               preprocess=False):

        if preprocess:
            (
                verts,
                faces_uvs,
                verts_uvs,
                verts_idx,
                texture_image,
                transform_head,
                transform_camera,
                focal,
                princpt,
            ) = self._jax2torch(verts, faces_uvs, verts_uvs, verts_idx,
                                texture_image, transform_head,
                                transform_camera, focal, princpt)

        tex = Textures(verts_uvs=verts_uvs,
                       faces_uvs=faces_uvs,
                       maps=texture_image)

        verts_head = self._linear_transform(verts, transform_head)
        verts_cam = self._linear_transform(verts_head, transform_camera)

        meshes = Meshes(verts=self._split_batch_to_list(verts_cam),
                        faces=self._split_batch_to_list(verts_idx),
                        textures=tex.to(self.device))

        cameras = PerspectiveCameras(device=self.device,
                                     focal_length=-focal,
                                     principal_point=princpt,
                                     in_ndc=False,
                                     image_size=(self.render_size, ))

        raster_settings = RasterizationSettings(
            image_size=[self.image_size[0], self.image_size[1]],
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

        return images