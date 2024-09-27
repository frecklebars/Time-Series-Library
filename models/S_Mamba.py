import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted

from contrastive.augmentation import RandomAUG, AutoAUG

from mamba_ssm import Mamba

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.is_pretraining = configs.is_pretraining
        # self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if configs.is_pretraining:
            # self.augmenter = RandomAUG(configs) # TODO get an augmentation strategy parameter
            self.augmenter = AutoAUG(configs) 
            # self.contrastive_projector = nn.Linear(configs.d_model * configs.enc_in, configs.pre_out, bias=True)
            self.contrastive_projector = nn.Linear(configs.d_model, configs.pre_out, bias=True) 
            


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def contrastive_pretrain(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, N = x_enc.shape

        # B L N -> B L N
        x_enc_aug_1, _ = self.augmenter(x_enc)
        x_enc_aug_2, _ = self.augmenter(x_enc)

        # B L N -> B N E
        x_enc_aug_1 = self.enc_embedding(x_enc_aug_1, x_mark_enc)
        x_enc_aug_2 = self.enc_embedding(x_enc_aug_2, x_mark_enc)

        # B N E -> B N E
        x_enc_aug_1_out, _ = self.encoder(x_enc_aug_1, attn_mask=None)
        x_enc_aug_2_out, _ = self.encoder(x_enc_aug_2, attn_mask=None)

        # x_enc_aug_1_out = x_enc_aug_1_out.permute(0, 2, 1)[:, :, :N]
        # x_enc_aug_2_out = x_enc_aug_2_out.permute(0, 2, 1)[:, :, :N]

        # x_enc_aug_1_out = x_enc_aug_1_out.permute(0, 2, 1).reshape(x_enc_aug_1_out.shape[0], x_enc_aug_1_out.shape[1] * x_enc_aug_1_out.shape[2])
        # x_enc_aug_2_out = x_enc_aug_2_out.permute(0, 2, 1).reshape(x_enc_aug_2_out.shape[0], x_enc_aug_2_out.shape[1] * x_enc_aug_2_out.shape[2])

        # B N E -> B N S
        dec_aug_1_out = self.contrastive_projector(x_enc_aug_1_out)[:, :N, :]
        dec_aug_2_out = self.contrastive_projector(x_enc_aug_2_out)[:, :N, :]

        # B N S -> B S
        B, N, S = dec_aug_1_out.shape
        dec_aug_1_out = dec_aug_1_out.reshape(B*N, S)
        dec_aug_2_out = dec_aug_2_out.reshape(B*N, S)

        # dec_aug_1_out = dec_aug_1_out.unsqueeze(1).permute(0, 2, 1)
        # dec_aug_2_out = dec_aug_2_out.unsqueeze(1).permute(0, 2, 1)

        return dec_aug_1_out, dec_aug_2_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.is_pretraining and self.training:
            dec_out_1, dec_out_2 = self.contrastive_pretrain(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out_1, dec_out_2
        if self.task_name in ['short_term_forecast', 'long_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]