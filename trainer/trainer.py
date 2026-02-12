import torch
from tensorboardX import SummaryWriter
import time
from parse_config import cfg
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import torchvision
from tqdm import tqdm
from data_loader.loader_ara import ContentData
import torch.distributed as dist
import torch.nn.functional as F

class Trainer:
    def __init__(self, diffusion, unet, vae, criterion, optimizer, data_loader,
                 logs, valid_data_loader=None, device=None, ocr_model=None, ctc_loss=None):
        self.model = unet
        self.diffusion = diffusion
        self.vae = vae
        self.recon_criterion = criterion['recon']
        self.nce_criterion = criterion['nce']
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.tb_summary = SummaryWriter(logs['tboard'])
        self.save_model_dir = logs['model']
        self.save_sample_dir = logs['sample']
        self.ocr_model = ocr_model
        self.ctc_criterion = ctc_loss
        self.device = device

        # Track best model based on validation loss
        self.best_loss = float('inf')
        self.best_epoch = -1
      
    def _train_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device)

        # Extract string writer IDs for annotations
        wid_str = data.get('wid_str', None)

        # vae encode
        images = self.vae.encode(images).latent_dist.sample()
        images = images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(images.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(images, t)


        predicted_noise, high_nce_emb, low_nce_emb = self.model(x_t, t, style_ref, laplace_ref, content_ref, tag='train')
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss values for return
        loss_values = {
            'recon': recon_loss.item(),
            'high_nce': high_nce_loss.item(),
            'low_nce': low_nce_loss.item(),
            'total': loss.item()
        }

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(), "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)

            # Print losses to terminal every 10 steps
            if step % 10 == 0:
                print(f"[Step {step}] recon: {recon_loss.item():.4f} | high_nce: {high_nce_loss.item():.4f} | low_nce: {low_nce_loss.item():.4f} | total: {loss.item():.4f}")

            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

        return loss_values

    def _finetune_iter(self, data, step, pbar):
        self.model.train()
        # prepare input

        images, style_ref, laplace_ref, content_ref, wid, target, target_lengths = data['img'].to(self.device), \
            data['style'].to(self.device), \
            data['laplace'].to(self.device), \
            data['content'].to(self.device), \
            data['wid'].to(self.device), \
            data['target'].to(self.device), \
            data['target_lengths'].to(self.device)
        
        # Extract string writer IDs for annotations
        wid_str = data.get('wid_str', None)
        
        # vae encode
        latent_images = self.vae.encode(images).latent_dist.sample()
        latent_images = latent_images * 0.18215


        # forward
        t = self.diffusion.sample_timesteps(latent_images.shape[0], finetune=True).to(self.device)
        x_t, noise = self.diffusion.noise_images(latent_images, t)
        
        x_start, predicted_noise, high_nce_emb, low_nce_emb = self.diffusion.train_ddim(self.model, x_t, style_ref, laplace_ref,
                                                        content_ref, t, sampling_timesteps=5)
 
        # calculate loss
        recon_loss = self.recon_criterion(predicted_noise, noise)
        rec_out = self.ocr_model(x_start)
        input_lengths = torch.IntTensor(x_start.shape[0]*[rec_out.shape[0]])
        ctc_loss = self.ctc_criterion(F.log_softmax(rec_out, dim=2), target, input_lengths, target_lengths)
        high_nce_loss = self.nce_criterion(high_nce_emb, labels=wid)
        low_nce_loss = self.nce_criterion(low_nce_emb, labels=wid)
        loss = recon_loss + high_nce_loss + low_nce_loss + 0.01*ctc_loss

        # backward and update trainable parameters
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.SOLVER.GRAD_L2_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
        self.optimizer.step()

        if dist.get_rank() == 0:
            # log file
            loss_dict = {"reconstruct_loss": recon_loss.item(),
                         "high_nce_loss": high_nce_loss.item(),
                         "low_nce_loss": low_nce_loss.item(),
                         "ctc_loss": ctc_loss.item()}
            self.tb_summary.add_scalars("loss", loss_dict, step)

            # Print all losses to terminal every 10 steps
            if step % 10 == 0:
                print(f"[Finetune Step {step}] recon: {recon_loss.item():.4f} | high_nce: {high_nce_loss.item():.4f} | low_nce: {low_nce_loss.item():.4f} | ctc: {ctc_loss.item():.4f} | total: {loss.item():.4f}")

            self._progress(recon_loss.item(), pbar)

        del data, loss
        torch.cuda.empty_cache()

    def _save_images(self, images, path, writer_ids=None):
        # Create grid of images
        grid = torchvision.utils.make_grid(images)
        im = torchvision.transforms.ToPILImage()(grid)
        
        # If we have writer IDs, add them as text
        if writer_ids is not None:
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()  # Use default font for simplicity
            
            # Calculate grid dimensions
            n_cols = min(8, images.shape[0])  # Default nrow in make_grid is 8
            n_rows = (images.shape[0] + n_cols - 1) // n_cols
            single_img_w = images.shape[3]
            single_img_h = images.shape[2]
            
            # Add text for each image
            for idx, wid in enumerate(writer_ids):
                row = idx // n_cols
                col = idx % n_cols
                x = col * (single_img_w + 2)  # +2 for padding in make_grid
                y = row * (single_img_h + 2)
                # Draw white background for better visibility
                text = str(wid)
                bbox = draw.textbbox((x, y), text, font=font)
                draw.rectangle(bbox, fill='white')
                # Draw text
                draw.text((x, y), text, fill='black', font=font)
        
        im.save(path)
        return im

    @torch.no_grad()
    def _valid_iter(self, epoch):
        if dist.get_rank() == 0:
            print(f'  Loading validation dataset ({len(self.valid_data_loader)} batches)...')
        self.model.eval()

        # use the first batch of dataloader in all validations for better visualization comparisons
        test_loader_iter = iter(self.valid_data_loader)
        test_data = next(test_loader_iter)

        # Prepare inputs: move to device.
        images = test_data['img'].to(self.device)
        style_ref = test_data['style'].to(self.device)
        laplace_ref = test_data['laplace'].to(self.device)
        content_ref = test_data['content'].to(self.device)
        writer_ids = test_data.get('wid_str', None) or [str(w.item()) for w in test_data['wid']]

        load_content = ContentData()
        # Define a fixed set of Urdu words for validation (from dataset vocabulary)
        texts = [
            'جمہویریہ',    # Republic
            'تابوت',       # Coffin
            'بحری',        # Naval
            'سیارچہ',      # Asteroid
            'نگہداشت',     # Care
            'اشعار',       # Poems
            'تحصیل'        # District
        ]
        if dist.get_rank() == 0:
            print(f'  Generating images for {len(texts)} texts')
        for idx, text in enumerate(texts):
            rank = dist.get_rank()
            # Get content glyphs for the text - generate only 1 image per sentence
            text_ref = load_content.get_content(text)
            text_ref = text_ref.to(self.device).unsqueeze(0)  # Add batch dimension for single image
            # Use only the first style reference for generation
            single_style = style_ref[0:1]  # Take first style only
            single_laplace = laplace_ref[0:1]  # Take first laplace only

            # Calculate width based on text length to avoid cutoff
            # Each character needs ~16 pixels in latent space, with some padding
            text_length = len(text)
            estimated_width = min(text_length * 2 + 4, 128)  # Cap at 128 (1024px in image space)
            latent_width = estimated_width

            x = torch.randn((1, 4, single_style.shape[2]//8, latent_width)).to(self.device)
            preds = self.diffusion.ddim_sample(self.model, self.vae, 1, x, single_style, single_laplace, text_ref)
            # Save single image (no grid needed)
            # Use index instead of text in filename to avoid Urdu encoding issues
            out_path = os.path.join(self.save_sample_dir, f"epoch{epoch+1}_val{idx+1}.png")
            self._save_images(preds, out_path, writer_ids=None)

            if dist.get_rank() == 0:
                print(f"  [{idx+1}/{len(texts)}] Saved: {os.path.basename(out_path)}")

        if dist.get_rank() == 0:
            print(f"  ✓ All validation images saved to: {self.save_sample_dir}")

    def train(self):
        """start training iterations"""
        start_time = time.time()
        for epoch in range(cfg.SOLVER.EPOCHS):
            epoch_start_time = time.time()
            self.data_loader.sampler.set_epoch(epoch)

            # Track losses for epoch summary
            epoch_losses = {'recon': [], 'high_nce': [], 'low_nce': [], 'total': []}

            if dist.get_rank() == 0:
                print(f"\n{'='*70}")
                print(f"Epoch {epoch+1}/{cfg.SOLVER.EPOCHS} | Process {dist.get_rank()}")
                print(f"{'='*70}")
            dist.barrier()
            if dist.get_rank() == 0:
                pbar = tqdm(self.data_loader, leave=False, desc=f"Epoch {epoch+1}")
            else:
                pbar = self.data_loader

            for step, data in enumerate(pbar):
                total_step = epoch * len(self.data_loader) + step
                if self.ocr_model is not None:
                    self._finetune_iter(data, total_step, pbar)
                    if (total_step+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (total_step+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                        if dist.get_rank() == 0:
                            self._save_checkpoint(total_step)
                    else:
                        pass
                    if self.valid_data_loader is not None:
                        if (total_step+1) > cfg.TRAIN.VALIDATE_BEGIN  and (total_step+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                            self._valid_iter(total_step)
                        else:
                            pass
                else:
                    loss_vals = self._train_iter(data, total_step, pbar)
                    if dist.get_rank() == 0:
                        epoch_losses['recon'].append(loss_vals['recon'])
                        epoch_losses['high_nce'].append(loss_vals['high_nce'])
                        epoch_losses['low_nce'].append(loss_vals['low_nce'])
                        epoch_losses['total'].append(loss_vals['total'])

            epoch_time = time.time() - epoch_start_time

            # Calculate average loss for this epoch
            avg_total_loss = None
            if dist.get_rank() == 0 and len(epoch_losses['total']) > 0:
                avg_total_loss = sum(epoch_losses['total']) / len(epoch_losses['total'])

            if (epoch+1) > cfg.TRAIN.SNAPSHOT_BEGIN and (epoch+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                if dist.get_rank() == 0:
                    print(f"\n[Checkpoint] Saving model at epoch {epoch+1}")
                    self._save_checkpoint(epoch, avg_total_loss)
                else:
                    pass
            if self.valid_data_loader is not None:
                if (epoch+1) > cfg.TRAIN.VALIDATE_BEGIN  and (epoch+1) % cfg.TRAIN.VALIDATE_ITERS == 0:
                    if dist.get_rank() == 0:
                        print(f"\n[Validation] Generating sample images for epoch {epoch+1}")
                    self._valid_iter(epoch)
            else:
                pass

            if dist.get_rank() == 0:
                pbar.close()

                # Print epoch summary with average losses
                if len(epoch_losses['total']) > 0:
                    avg_recon = sum(epoch_losses['recon']) / len(epoch_losses['recon'])
                    avg_high = sum(epoch_losses['high_nce']) / len(epoch_losses['high_nce'])
                    avg_low = sum(epoch_losses['low_nce']) / len(epoch_losses['low_nce'])
                    avg_total = sum(epoch_losses['total']) / len(epoch_losses['total'])
                    print(f"\n[Epoch {epoch+1} Summary]")
                    print(f"  Avg Losses - recon: {avg_recon:.4f} | high_nce: {avg_high:.4f} | low_nce: {avg_low:.4f} | total: {avg_total:.4f}")

                elapsed_time = time.time() - start_time
                eta = elapsed_time / (epoch + 1) * (cfg.SOLVER.EPOCHS - epoch - 1)
                print(f"  Time: {epoch_time:.2f}s | Total: {elapsed_time/60:.2f}min | ETA: {eta/60:.2f}min")

    def _progress(self, loss, pbar):
        pbar.set_postfix(mse='%.6f' % (loss))

    def _save_checkpoint(self, epoch, avg_loss=None):
        checkpoint_path = os.path.join(self.save_model_dir, str(epoch)+'-'+"ckpt.pt")
        torch.save(self.model.module.state_dict(), checkpoint_path)
        print(f"  ✓ Checkpoint saved: {os.path.basename(checkpoint_path)}")

        # Save best model if this is the best epoch so far
        if avg_loss is not None and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_epoch = epoch
            best_path = os.path.join(self.save_model_dir, "best_model.pt")
            torch.save(self.model.module.state_dict(), best_path)
            print(f"  ✓ Best model saved! (epoch {epoch+1}, loss: {avg_loss:.4f})")

        # Auto-delete old checkpoints to save space (keep only last 3)
        all_checkpoints = sorted([f for f in os.listdir(self.save_model_dir) if f.endswith('-ckpt.pt')])
        if len(all_checkpoints) > 3:
            # Delete oldest checkpoints, keep only last 3
            for old_ckpt in all_checkpoints[:-3]:
                old_path = os.path.join(self.save_model_dir, old_ckpt)
                try:
                    os.remove(old_path)
                    print(f"  Deleted old checkpoint: {old_ckpt}")
                except:
                    pass