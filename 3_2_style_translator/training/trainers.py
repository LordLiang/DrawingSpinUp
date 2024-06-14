import os
import time
import six
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from training.custom_transforms import to_image_space
from training.data import DatasetPatches_M,  DatasetFullImages


class Trainer(object):
    def __init__(self,
                 data_root, trainer_config, 
                 opt_discriminator, opt_generator, model_logger,
                 perception_loss_model, perception_loss_weight,
                 use_pos, use_edge, device
                 ):
        
        training_dataset = DatasetPatches_M(os.path.join(data_root, 'keyframe'),
                                            trainer_config['dir_post'],
                                            trainer_config['patch_size'], use_pos, use_edge)
        self.train_loader = torch.utils.data.DataLoader(training_dataset, 
                                                        trainer_config['batch_size'], 
                                                        shuffle=False,
                                                        num_workers=trainer_config['num_workers'], 
                                                        drop_last=True)

        self.opt_discriminator = opt_discriminator
        self.opt_generator = opt_generator

        # set loss functions
        self.reconstruction_criterion = getattr(torch.nn, trainer_config['reconstruction_criterion'])()
        self.adversarial_criterion = getattr(torch.nn, trainer_config['adversarial_criterion'])()
        self.reconstruction_weight=trainer_config['reconstruction_weight']
        self.adversarial_weight=trainer_config['adversarial_weight']

        self.model_logger = model_logger
        self.training_log = {}
        self.log_interval = trainer_config['log_interval']

        self.perception_loss_weight = perception_loss_weight
        self.perception_loss_model = perception_loss_model

        self.use_adversarial_loss = False
        self.use_image_loss = trainer_config['use_image_loss']
        self.device = device

        self.data_root = data_root
        self.testing_name_list = trainer_config['testing_name_list']
        self.use_pos = use_pos
        self.use_edge = use_edge


    def run_discriminator(self, discriminator, images):
        return discriminator(images)

    def compute_discriminator_loss(self, generator, discriminator, batch):
        generated = generator(batch['pre'])
        fake = self.apply_mask(generated, batch, 'pre_mask')

        fake_labels, _ = self.run_discriminator(discriminator, fake.detach())

        true = self.apply_mask(batch['already'], batch, 'already_mask')
        true_labels, _ = self.run_discriminator(discriminator, true)

        discriminator_loss = self.adversarial_criterion(fake_labels, self.zeros_like(fake_labels)) + \
                             self.adversarial_criterion(true_labels, self.ones_like(true_labels))

        return discriminator_loss

    def compute_generator_loss(self, generator, discriminator, batch, use_gan, use_mask):
        image_loss = 0
        perception_loss = 0
        adversarial_loss = 0

        generated = generator(batch['pre'])

        if use_mask:
            generated = generated * batch['mask']
            batch['post'] = batch['post'] * batch['mask']

        if self.use_image_loss:
            if generated[0][0].shape != batch['post'][0][0].shape:
                if ((batch['post'][0][0].shape[0] - generated[0][0].shape[0]) % 2) != 0:
                    raise RuntimeError("batch['post'][0][0].shape[0] - generated[0][0].shape[0] must be even number")
                if generated[0][0].shape[0] != generated[0][0].shape[1] or batch['post'][0][0].shape[0] != batch['post'][0][0].shape[1]:
                    raise RuntimeError("And also it is expected to be exact square ... fix it if you want")
                boundary_size = int((batch['post'][0][0].shape[0] - generated[0][0].shape[0]) / 2)
                cropped_batch_post = batch['post'][:, :, boundary_size: -1*boundary_size, boundary_size: -1*boundary_size]
                image_loss = self.reconstruction_criterion(generated, cropped_batch_post)
            else:
                image_loss = self.reconstruction_criterion(generated, batch['post'])

        if self.perception_loss_model is not None:
            _, fake_features = self.perception_loss_model(generated)
            _, target_features = self.perception_loss_model(Variable(batch['post'], requires_grad=False))
            perception_loss = ((fake_features - target_features) ** 2).mean()


        if self.use_adversarial_loss and use_gan:
            fake = self.apply_mask(generated, batch, 'pre_mask')
            fake_smiling_labels, _ = self.run_discriminator(discriminator, fake)
            adversarial_loss = self.adversarial_criterion(fake_smiling_labels, self.ones_like(fake_smiling_labels))

        return image_loss, perception_loss, adversarial_loss, generated


    def train(self, generator, discriminator, epochs, result_folder, starting_batch_num):
        self.use_adversarial_loss = discriminator is not None

        batch_num = starting_batch_num
        save_num = 0

        start = time.time()
        for epoch in range(epochs):
            np.random.seed()
            for i, batch in enumerate(self.train_loader):
                # just sets the models into training mode (enable BN and DO)
                [m.train() for m in [generator, discriminator] if m is not None]
                batch = {k: batch[k].to(self.device) if isinstance(batch[k], torch.Tensor) else batch[k]
                         for k in batch.keys()}

                # train discriminator
                if self.use_adversarial_loss:
                    self.opt_discriminator.zero_grad()
                    discriminator_loss = self.compute_discriminator_loss(generator, discriminator, batch)
                    discriminator_loss.backward()
                    self.opt_discriminator.step()

                # train generator
                self.opt_generator.zero_grad()

                g_image_loss, g_perc_loss, g_adv_loss, _ = self.compute_generator_loss(generator, discriminator, batch, use_gan=True, use_mask=False)

                generator_loss = self.reconstruction_weight * g_image_loss + \
                                 self.perception_loss_weight * g_perc_loss + \
                                 self.adversarial_weight * g_adv_loss

                generator_loss.backward()

                self.opt_generator.step()

                # log losses
                current_log = {key: value.item() for key, value in six.iteritems(locals()) if
                               'loss' in key and isinstance(value, Variable)}

                self.add_log(current_log)

                batch_num += 1

                if batch_num % self.log_interval == 0 or batch_num == 1:
                    eval_start = time.time()
                    generator.eval()
                    self.test_on_full_image(generator, result_folder)
                    self.flush_scalar_log(batch_num, time.time() - start)
                    self.model_logger.save(generator, save_num, True)
                    save_num += 1
                    print(f"Eval of batch: {batch_num} took {(time.time() - eval_start)}", flush=True)

        self.model_logger.save(generator, 99999, True)
    
    # Accumulates the losses
    def add_log(self, log):
        for k, v in log.items():
            if k in self.training_log:
                self.training_log[k] += v
            else:
                self.training_log[k] = v

    # Divide the losses by log_interval and print'em
    def flush_scalar_log(self, batch_num, took):
        log = "[%d]" % batch_num
        for key in sorted(self.training_log.keys()):
            log += " [%s] % 7.4f" % (key, self.training_log[key] / self.log_interval)

        log += ". Took {}".format(took)
        print(log, flush=True)
        self.training_log = {}

    # Test the intermediate model on data from _gen folder
    def test_on_full_image(self, generator, result_folder, save_alpha=False):
        for test_name in self.testing_name_list:
            data_root = os.path.join(self.data_root, test_name)
            dataset = DatasetFullImages(data_root, self.use_pos, self.use_edge)
            imloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, num_workers=1, drop_last=False)  # num_workers=4
            
            with torch.no_grad():
                for i, batch in enumerate(imloader):
                    batch = {k: batch[k].to(self.device) if isinstance(batch[k], torch.Tensor) else batch[k]
                            for k in batch.keys()}
                    gan_output = generator(batch['pre'])[0]
                    
                    gt_test_ganoutput_path = os.path.join(data_root, result_folder)
                    os.makedirs(gt_test_ganoutput_path, exist_ok=True)

                    img = to_image_space(gan_output.cpu().numpy()).transpose(1, 2, 0)
                    if save_alpha:
                        alpha = (batch['pre_mask'][0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        img = np.concatenate((img, alpha), 2)
                    Image.fromarray(img).save(os.path.join(gt_test_ganoutput_path, batch['file_name'][0]))

    def apply_mask(self, x, batch, mask_key):
        if mask_key in batch:
            mask = Variable(batch[mask_key].expand(x.size()), requires_grad=False)
            return x * mask
        return x

    def ones_like(self, x):
        return torch.ones_like(x).to(self.device)

    def zeros_like(self, x):
        return torch.zeros_like(x).to(self.device)
