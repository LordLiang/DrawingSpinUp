import numpy as np
import mcubes
import cv2

import torch
import torch.nn as nn
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from nerfacc import ContractionType

from instant_nsr import models
from instant_nsr.models.base import BaseModel
from instant_nsr.models.utils import scale_anything, get_activation, cleanup, chunk_batch
from instant_nsr.models.network_utils import get_encoding, get_mlp
from instant_nsr.utils.mesh_utils import remesh
from instant_nsr.systems.utils import update_module_step


def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


class MarchingCubeHelper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resolution = config.isosurface.resolution
        self.remeshing = config.remeshing
        self.face_count = config.face_count
        self.points_range = (0, 1)
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution), torch.linspace(*self.points_range, self.resolution)
            x, y, z = torch.meshgrid(x, y, z, indexing='ij')
            verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts
        return self.verts

    def forward(self, level, threshold=0., fine_stage=False, front_mask=None):
        level = level.float().view(self.resolution, self.resolution, self.resolution)
        if front_mask is not None:
            front_mask = cv2.resize(front_mask, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
            front_mask = np.tile(front_mask[:, None ,:], (1, self.resolution, 1))
            binary_value = (level.numpy() <= 0) * (front_mask > 127)
            value = mcubes.smooth(binary_value)
        else:
            binary_value = level.numpy() <= 0
            value = mcubes.smooth(binary_value)

        verts, faces = mcubes.marching_cubes(value, threshold)
        verts = verts / (self.resolution - 1.)
        if fine_stage and self.remeshing:
                verts, faces = remesh(verts, faces, self.face_count)

        return {
            'verts': torch.from_numpy(verts),
            'faces': faces
        }
    

class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            self.helper = MarchingCubeHelper(self.config)
        self.radius = self.config.radius
        self.contraction_type = None # assigned in system

    def forward_level(self, points):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax, fine_stage=False, front_mask=None):
        def batch_func(x):
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv
        if front_mask is not None:
            size = front_mask.shape[0] / 2
            x_min, x_max = torch.floor(vmin[0]*size+size).int().numpy(), torch.ceil(vmax[0]*size+size).int().numpy()
            z_min, z_max = torch.floor(vmin[2]*size+size).int().numpy(), torch.ceil(vmax[2]*size+size).int().numpy()
            front_mask = front_mask[x_min:x_max, z_min:z_max]
            
        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices())
        mesh = self.helper(level, threshold=self.config.isosurface.threshold, fine_stage=fine_stage, front_mask=front_mask)
        mesh['verts'] = torch.stack([
            scale_anything(mesh['verts'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['verts'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['verts'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self, front_mask=None):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius))
        vmin, vmax = mesh_coarse['verts'].amin(dim=0), mesh_coarse['verts'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_, True, front_mask)
        return mesh_fine 


@models.register('volume-sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        encoding = get_encoding(3, self.config.xyz_encoding_config)
        network = get_mlp(encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.encoding, self.network = encoding, network
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        # the actual value used in training
        # will update at certain steps if finite_difference_eps="progressive"
        self._finite_difference_eps = None
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
                
                out = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims).float()
                sdf, feature = out[...,0], out
                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True
                        )[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  

                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        sdf = self.network(self.encoding(points.view(-1, 3))).view(*points.shape[:-1], self.n_output_dims)[...,0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        if self.grad_type == 'finite_difference':
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(f"Update finite_difference_eps to {grid_size}")
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")


