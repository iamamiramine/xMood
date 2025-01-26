import os
import pickle

import math
import random
from pathlib import Path

import torch
import torch.optim
import torch.nn as nn
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

from lightning.pytorch import LightningModule

from persistence.dataloader.repositories.dataloader_repository import save_async, async_load
from application.encoder.helpers.vocab_helper import get_bars, mask_bar_tokens, get_bos_eos_events
from domain.constants.paths_constants import LATENTS_PATH, PROCESSED_PATH
from domain.constants.encoder.token_constants import (
    PAD_TOKEN,
    MASK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)
from application.feature_extraction.models.vq_ema_model import VectorQuantizeEMA
from application.encoder.models.vocab_model import RemiVocab


class VqVaeModule(LightningModule):
    def __init__(
        self,
        dataset_name="",
        d_model=512,
        context_size=256,
        n_codes=1024,
        n_groups=2,
        d_latent=1024,
        lr=1e-4,
        lr_schedule="sqrt_decay",
        warmup_steps=1000,
        max_steps=10000,
        encoder_layers=6,
        decoder_layers=6,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        windowed_attention_pr=0.0,
        max_lookahead=4,
        disable_vq=False,
        accumulate_grad_batches=1,
        max_positions=1024,
        automatic_optimization=False,
        beta=0.02,
        cycle_length=2000,
        position_embedding_type="relative_key_query",
        num_attention_heads=8,
        decay=0.995,
        eps=1e-4,
        restart_threshold=0.99,
        vocab=RemiVocab(),
    ):
        super().__init__()
        self.automatic_optimization = automatic_optimization
        self.accumulate_grad_batches = accumulate_grad_batches

        self.d_model = d_model  #  size of the embeddings and the hidden layers throughout the encoder and decoder
        self.context_size = context_size  # context size for the attention mask
        self.n_codes = n_codes  # number of codes in the codebook
        self.n_groups = n_groups  # number of groups in the codebook
        self.d_latent = d_latent  #  size of the latent space

        self.lr = lr  # learning rate
        self.lr_schedule = lr_schedule  # learning rate schedule
        self.warmup_steps = warmup_steps  # number of steps to warm up the learning rate
        self.max_steps = max_steps  # maximum number of steps to train
        self.windowed_attention_pr = windowed_attention_pr  # probability of using windowed attention
        self.max_lookahead = max_lookahead  # maximum lookahead for windowed attention
        self.disable_vq = disable_vq  # whether to disable the VQ-VAE

        self.vocab = vocab  # vocabulary
        self.dataset_name = dataset_name  # name of the dataset

        self.beta = beta  # beta parameter for the VQ-VAE
        self.cycle_length = cycle_length  # cycle length for the learning rate schedule

        self.decay = decay  # decay rate for the codebook
        self.eps = eps  # epsilon for the codebook
        self.restart_threshold = restart_threshold  # restart threshold for the codebook

        self.pad_token = self.vocab.to_i(PAD_TOKEN)
        self.bos_token = self.vocab.to_i(BOS_TOKEN)
        self.eos_token = self.vocab.to_i(EOS_TOKEN)
        self.mask_token = self.vocab.to_i(MASK_TOKEN)

        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=encoder_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=encoder_ffn_dim,
            max_position_embeddings=max_positions,
            position_embedding_type=position_embedding_type,
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.d_model,
            num_hidden_layers=decoder_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=decoder_ffn_dim,
            max_position_embeddings=max_positions,
            position_embedding_type=position_embedding_type,
        )

        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.transformer = EncoderDecoderModel(config)
        self.transformer.config.decoder.is_decoder = True
        self.transformer.config.decoder.add_cross_attention = True
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder

        self.in_layer = nn.Embedding(len(self.vocab), self.d_model)
        self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)

        self.vq_embed = VectorQuantizeEMA(
            self.d_latent,
            self.n_codes,
            self.n_groups,
            self.decay,
            self.eps,
            self.restart_threshold,
        )
        self.pooling = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.unpooling = nn.Linear(self.d_latent, self.d_model, bias=False)
        self.attention_proj = nn.Linear(self.d_model, self.d_model)

        self.rec_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token)

        self.save_hyperparameters()

    def forward(self, x, y=None, latent=None, use_windowed_attention=False, return_latent_logits=False):
        if y is None:
            y = x.clone().detach()

        # VQ-VAE
        if latent is None:
            encoder_out = self.encode(x)
            latent = encoder_out["z"]

        logits = self.decode(x, latent, use_windowed_attention)
        return {"logits": logits, **encoder_out}

    def embed(self, x):
        return self.in_layer(x)

    def encode(self, x):
        x_emb = self.embed(x)

        # Shape of out: (batch_size, seq_len, d_model)
        out = self.encoder(inputs_embeds=x_emb, output_hidden_states=True)
        hidden = out.pooler_output
        # Shape of z_e: (batch_size, d_model * n_groups)
        z_e = self.pooling(hidden)

        if self.disable_vq:
            # AE baseline
            return {"z": z_e}
        else:
            # VQ-VAE
            # Shape of z_q: (batch_size, d_model * n_groups)
            dist = self.trainer.accelerator.training_type_plugin if self.training else None  # TODO: Remove training_type_plugin
            return self.vq_embed(z_e, dist=dist)

    def decode(self, x, latent, use_windowed_attention=False):
        # Shape of latent: (batch_size, n_groups, d_model)
        x_emb = self.embed(x)
        seq_len = x_emb.size(1)

        # Shape of h0: (batch_size, d_model)
        h0 = self.unpooling(latent)

        # Make model decoder-only by fixing h0
        # h0 = torch.zeros_like(h0)

        # Strategy 1: Add latent embeddings to input embeddings
        x_emb += h0.unsqueeze(1).repeat(1, seq_len, 1)

        # Strategy 2: Use latent embedding in cross-attention
        x_attention = self.attention_proj(h0.unsqueeze(1).repeat(1, self.context_size, 1))

        # Relative pos. embeddings need source and target to be of the same length
        # -> prevents einsum shape mismatch error
        padding = torch.zeros_like(x_attention)
        padding[:, : x_emb.size(1)] = x_emb
        x_emb = padding

        if self.training or use_windowed_attention:
            attention_mask = self.rand_attention_mask(x)
        else:
            attention_mask = self.get_attention_mask(x)
        padding = torch.zeros((x.size(0), self.context_size, self.context_size), device=self.device, dtype=torch.int)
        padding[:, : attention_mask.size(1), : attention_mask.size(2)] = attention_mask
        attention_mask = padding

        out = self.decoder(inputs_embeds=x_emb, encoder_hidden_states=x_attention, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, :seq_len]
        logits = self.out_layer(hidden).contiguous()

        return logits

    def get_loss(self, batch, windowed_attention_pr=None):
        if windowed_attention_pr is None:
            windowed_attention_pr = self.windowed_attention_pr
        use_windowed_attention = True if random.random() < windowed_attention_pr else False

        x = batch["input_ids"]
        labels = batch["labels"]

        out = self.forward(
            x,
            y=labels,
            use_windowed_attention=use_windowed_attention,
        )

        logits = out["logits"]
        # Reshape logits to: (batch_size * seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))
        # Reshape labels to: (batch_size * seq_len)
        labels = labels.view(-1)

        rec_loss = self.rec_loss(logits, labels)

        if self.disable_vq:
            loss = rec_loss
        else:
            diff = out["diff"]
            loss = rec_loss + self.beta * diff

        return {"loss": loss, "rec_loss": rec_loss, **out}

    # Manual Optimization: https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    # https://stackoverflow.com/questions/77325527/relationship-between-gradient-clipping-and-gradient-accumulation
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()  # Manual Optimization

        metrics = self.get_loss(batch)  # Get Loss
        log_metrics = {key: metrics[key].detach() for key in ["loss", "rec_loss", "diff", "avg_usage", "usage", "entropy"] if key in metrics}
        # self.log('train', log_metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        batch_loss = metrics["loss"] / self.accumulate_grad_batches

        self.manual_backward(batch_loss)

        # accumulate and clip gradients of N batches
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
            opt.zero_grad()

        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        log_metrics = {key: metrics[key].detach() for key in ["rec_loss", "diff", "avg_usage", "usage", "entropy"] if key in metrics}

        # Compute perplexity
        x, y = batch["input_ids"], batch["labels"]
        pad_token_id = self.vocab.to_i(PAD_TOKEN)
        logits = metrics["logits"]
        log_pr = logits.log_softmax(dim=-1)
        log_pr[y == pad_token_id] = 0  # log(pr) = log(1) for padding
        log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)
        t = (y != pad_token_id).sum(dim=-1)
        ppl = (-log_pr.sum(dim=1) / t).exp().mean()
        log_metrics["ppl"] = ppl.detach()

        # self.log("valid", log_metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # Log loss separately for model checkpoint monitor
        self.log(
            "valid_loss",
            metrics["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return metrics["loss"]

    def test_step(self, batch, batch_idx):
        metrics = self.get_loss(batch)
        return metrics["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = self.trainer.global_step

        # # beta is increased for C*R steps and then held constant for C*(1-R) steps
        # if step/2 >= self.cycle_length:
        #     self.cycle_length *= 2
        # C = self.cycle_length # cycle length
        # R = 1000 # restart steps
        # b_min, b_max = 0.0, 0.1
        # t = max(0, min(1, (step % C) / R))
        # self.beta = b_min*(1 - t) + b_max*t
        # self.log('beta', self.beta, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

    def configure_optimizers(self):
        # set LR to 1, scale with LambdaLR scheduler
        optimizer = torch.optim.AdamW(self.parameters(), lr=1, weight_decay=0.01)

        if self.lr_schedule == "sqrt_decay":
            # constant warmup, then 1/sqrt(n) decay starting from the initial LR
            lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1) / self.warmup_steps))
        elif self.lr_schedule == "linear":
            # linear warmup, linear decay
            lr_func = lambda step: min(
                self.lr,
                self.lr * step / self.warmup_steps,
                self.lr * (1 - (step - self.warmup_steps) / self.max_steps),
            )
        elif self.lr_schedule == "cosine":
            # linear warmup, cosine decay to 10% of initial LR
            lr_func = lambda step: self.lr * min(
                step / self.warmup_steps,
                0.55 + 0.45 * math.cos(math.pi * (min(step, self.max_steps) - self.warmup_steps) / (self.max_steps - self.warmup_steps)),
            )
        else:
            # Use no lr scheduling
            lr_func = lambda step: self.lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
            }
        ]

    def rand_attention_mask(self, x, pr=0.2, max_size=None):
        if max_size == None:
            max_size = self.max_lookahead
        if max_size is not None and self.training and random.random() < pr:
            mask_size, k = random.randint(1, max_size), 0
        else:
            mask_size, k = 1, 1
        return self.get_attention_mask(x, mask_size=mask_size, k=k)

    def get_attention_mask(self, x, mask_size=1, k=1):
        batch_size, seq_len = x.shape[:2]

        # Standard self-attention mask for auto-regressive modelling
        tri_mask = torch.ones(
            (seq_len // mask_size + 1, seq_len // mask_size + 1),
            device=self.device,
            dtype=torch.int,
        )
        tri_mask = torch.triu(tri_mask, diagonal=k)
        tri_mask = (~tri_mask.bool()).int()
        # Create windowed self-attention mask, forcing the model to prefict farther into the future
        window_mask = tri_mask.repeat_interleave(mask_size, dim=0).repeat_interleave(mask_size, dim=1)[:seq_len, :seq_len]
        # First token needs to be always visible
        window_mask[:, 0] = 1

        return window_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = {}
        events = batch["input_ids"]

        output_path = os.path.join(PROCESSED_PATH, self.dataset_name)

        for i, event_ids in enumerate(events):
            file = batch["files"][i]
            try:
                # Load existing processed data
                processed_data = async_load(output_path, file, "processed")

                # Skip if latents already exist
                if "latents" in processed_data:
                    continue

                events_dec = self.vocab.decode(event_ids)

                bars = get_bars(events_dec)
                mask_bar_tokens(events_dec, bar_token_mask=None)

                groups = [event_ids[start:end] for start, end in zip(bars[:-1], bars[1:])]
                groups.append(event_ids[bars[-1] :])

                bos, eos = get_bos_eos_events(self.vocab)
                bos = bos.to(self.device)
                eos = eos.to(self.device)

                latents = []
                codes = []

                for bar in groups:
                    x = torch.cat([bos, bar, eos])[: self.context_size].unsqueeze(0)
                    out = self.encode(x)
                    z, code = out["z"], out["codes"]
                    latents.append(z)
                    codes.append(code)

                latents = torch.cat(latents)
                codes = torch.cat(codes)

                # Update processed data with latents
                processed_data["latents"] = {
                    "latents": latents.cpu().numpy()[0],
                    "codes": codes.cpu().numpy()[0],
                }

                # Save back to processed file
                save_async(output_path, file, processed_data, "processed")

            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {str(e)}")

        return pred
