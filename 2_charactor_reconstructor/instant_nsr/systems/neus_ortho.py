import torch
import torch.nn.functional as F

from instant_nsr import systems
from instant_nsr.models.ray_utils import get_ortho_rays
from instant_nsr.systems.base import BaseSystem
from instant_nsr.systems.criterions import binary_cross_entropy, ranking_loss
from instant_nsr.utils.post_processing import save_mesh


@systems.register('ortho-neus-system')
class OrthoNeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_masks), size=(self.train_num_rays,), device=self.dataset.all_masks.device)
            else:
                index = torch.randint(0, len(self.dataset.all_masks), size=(1,), device=self.dataset.all_masks.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_masks.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_masks.device
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
                origins = self.dataset.origins[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
                origins = self.dataset.origins[index, y, x]
            rays_o, rays_d = get_ortho_rays(origins, directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            normal = self.dataset.all_normals_world[index, y, x].view(-1, self.dataset.all_normals_world.shape[-1]).to(self.rank)
            mask = self.dataset.all_masks[index, y, x].view(-1).to(self.rank)
            view_weights = self.dataset.view_weights[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
                origins = self.dataset.origins
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
                origins = self.dataset.origins[index][0]
            rays_o, rays_d = get_ortho_rays(origins, directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            normal = self.dataset.all_normals_world[index].view(-1, self.dataset.all_normals_world.shape[-1]).to(self.rank)
            mask = self.dataset.all_masks[index].view(-1).to(self.rank)
            view_weights = None

        cosines = self.cos(rays_d, normal)
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'normal': normal,
            'mask': mask,
            'cosines': cosines,
            'view_weights': view_weights
        })

    def training_step(self, batch, batch_idx):
        out = self(batch)

        cosines = batch['cosines']
        view_weights =  batch['view_weights']
        cosines[cosines > -0.1] = 0
        mask = ((batch['mask'] > 0) & (cosines < -0.1))

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        erros_rgb_mse = F.mse_loss(out['comp_rgb'][mask], batch['rgb'][mask], reduction='none')
        loss_rgb_mse = ranking_loss(erros_rgb_mse.sum(dim=1), 
                                    penalize_ratio=self.config.system.loss.rgb_p_ratio, type='mean')
        self.log('train/loss_rgb_mse', loss_rgb_mse, prog_bar=True, rank_zero_only=True)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb'][mask], batch['rgb'][mask], reduction='none')
        loss_rgb_l1 = ranking_loss(loss_rgb_l1.sum(dim=1),
                                    penalize_ratio=0.8)
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)    

        normal_errors = 1 - F.cosine_similarity(out['comp_normal'], batch['normal'], dim=1)
        if self.config.system.loss.geo_aware:
            normal_errors = normal_errors * torch.exp(cosines.abs()) / torch.exp(cosines.abs()).sum()
            loss_normal = ranking_loss(normal_errors[mask], 
                                    penalize_ratio=self.config.system.loss.normal_p_ratio, 
                                    extra_weights=view_weights[mask],
                                    type='sum')
        else:
            loss_normal = ranking_loss(normal_errors[mask], 
                                    penalize_ratio=self.config.system.loss.normal_p_ratio, 
                                    extra_weights=view_weights[mask],
                                    type='mean')    
        
        self.log('train/loss_normal', loss_normal, prog_bar=True, rank_zero_only=True)
        loss += loss_normal * self.C(self.config.system.loss.lambda_normal)       

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal, prog_bar=True, rank_zero_only=True)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['mask'].float(), reduction='none')
        loss_mask = ranking_loss(loss_mask, 
                                 penalize_ratio=self.config.system.loss.mask_p_ratio, 
                                 extra_weights=view_weights)
        self.log('train/loss_mask', loss_mask, prog_bar=True, rank_zero_only=True)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['random_sdf'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity, prog_bar=True, rank_zero_only=True)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)
        
        if self.C(self.config.system.loss.lambda_3d_normal_smooth) > 0:
            if "random_sdf_grad" not in out:
                raise ValueError(
                    "random_sdf_grad is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals_3d = out["random_sdf_grad"]
            normals_perturb_3d = out["normal_perturb"]
            loss_3d_normal_smooth = (normals_3d - normals_perturb_3d).abs().mean()
            self.log('train/loss_3d_normal_smooth', loss_3d_normal_smooth, prog_bar=True )
            loss += loss_3d_normal_smooth *  self.C(self.config.system.loss.lambda_3d_normal_smooth)  

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }         
    
    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        if self.trainer.is_global_zero:
            self.export()
    
    def export(self):
        save_name = f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}-{self.config.model.geometry.face_count}"
        if self.config.model.geometry.front_cutting:
            save_name += '_cut'
        if self.config.model.geometry.mesh_simplify:
            save_name += '_simpl' 
        if not self.config.export.vertex_coloring:
            save_name += '_mlp'
        self.config.export.save_name = save_name
        self.config.export.input_dir = self.config.dataset.input_dir
        mesh = self.model.export(self.config.export, self.dataset.front_mask)
        save_mesh(
            self.config.export,
            **mesh
        )        
