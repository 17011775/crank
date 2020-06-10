#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Differential VQVAE class

"""

import torch
import torch.nn as nn

from parallel_wavegan.models import ParallelWaveGANGenerator
from crank.net.module.vqvae2 import VQVAE2


class DiffVQVAE2(VQVAE2):
    def __init__(self, net_conf, spkr_size=0):
        super().__init__(net_conf, spkr_size)
        # self._construct_diff_decoder()
        self._construct_diff_decoder_multi()

    def forward(self, x, enc_h=None, dec_h=None, use_ema=True):
        x = x.transpose(1, 2)
        enc_h = enc_h.transpose(1, 2) if enc_h is not None else None
        dec_h = dec_h.transpose(1, 2) if dec_h is not None else None

        enc, spkr_cls = self.encode(x, enc_h=enc_h)
        enc_vqave, dec, emb_idxs, _, qidxs = self.decode(
            [e.clone() for e in enc], dec_h, use_ema=use_ema
        )
        _, diff_dec, _, _, _ = self.diff_decode_multi(
            [e.clone() for e in enc], dec_h, use_ema=use_ema
        )
        outputs = self.make_dict(enc_vqave, dec, emb_idxs, qidxs, spkr_cls)
        outputs.update({"decoded_diff": (diff_dec + x).transpose(1, 2)})
        return outputs

    def diff_forward(
        self, x, org_enc_h=None, org_dec_h=None, cv_enc_h=None, cv_dec_h=None
    ):
        x = x.transpose(1, 2)
        org_enc_h = org_enc_h.transpose(1, 2) if org_enc_h is not None else None
        org_dec_h = org_dec_h.transpose(1, 2) if org_dec_h is not None else None
        cv_dec_h = cv_dec_h.transpose(1, 2) if cv_dec_h is not None else None

        # vqvae
        enc, org_spkr_cls = self.encode(x, enc_h=org_enc_h)
        org_enc, org_dec, org_emb_idxs, _, org_qidxs = self.decode(
            [e.clone() for e in enc], org_dec_h, use_ema=True
        )

        # cv
        cv_enc, cv_dec, cv_emb_idxs, _, cv_qidxs = self.decode(
            [e.clone() for e in enc], cv_dec_h, use_ema=True
        )

        # diff
        diff_enc, diff_dec, diff_emb_idxs, _, diff_qidxs = self.diff_decode_multi(
            [e.clone() for e in enc], cv_dec_h, use_ema=True
        )

        # cv_enc, cv_dec, diff_dec, cv_emb_idxs, _, cv_qidxs = self.diff_decode(
        #     [e.clone() for e in enc], cv_dec_h, use_ema=False
        # )

        outputs = {
            "org": self.make_dict(
                org_enc, org_dec, org_emb_idxs, org_qidxs, org_spkr_cls
            ),
            "cv": self.make_dict(cv_enc, cv_dec, cv_emb_idxs, cv_qidxs, None),
            "diffcv": self.make_dict(
                diff_enc, diff_dec + x, diff_emb_idxs, diff_qidxs, None
            ),
        }
        return outputs

    # def diff_forward_cycle(
    #     self, x, org_enc_h=None, org_dec_h=None, cv_enc_h=None, cv_dec_h=None
    # ):
    #     x = x.transpose(1, 2)
    #     org_enc_h = org_enc_h.transpose(1, 2) if org_enc_h is not None else None
    #     org_dec_h = org_dec_h.transpose(1, 2) if org_dec_h is not None else None
    #     cv_enc_h = cv_enc_h.transpose(1, 2) if cv_enc_h is not None else None
    #     cv_dec_h = cv_dec_h.transpose(1, 2) if cv_dec_h is not None else None

    #     outputs = []
    #     for n in range(self.net_conf["n_cycles"]):
    #         # vqvae
    #         enc, org_spkr_cls = self.encode(x, enc_h=org_enc_h)
    #         org_enc, org_dec, org_emb_idxs, _, org_qidxs = self.decode(
    #             [e.clone() for e in enc], org_dec_h, use_ema=True
    #         )

    #         # cv and diff
    #         cv_enc, cv_dec, diff_dec, cv_emb_idxs, _, cv_qidxs = self.diff_decode(
    #             [e.clone() for e in enc], cv_dec_h, use_ema=False
    #         )

    #         _, cv_dec, _, _, _ = self.decode(
    #             [e.clone() for e in enc], cv_dec_h, use_ema=False
    #         )

    #         # reconstruct for cv
    #         enc, cv_spkr_cls = self.encode(cv_dec, enc_h=cv_enc_h)
    #         recon_enc, recon_dec, recon_emb_idxs, _, recon_qidxs = self.decode(
    #             enc, org_dec_h, use_ema=False
    #         )

    #         # reconstruct for diff
    #         enc, diffcv_spkr_cls = self.encode(diff_dec + x, enc_h=cv_enc_h)
    #         (
    #             diff_recon_enc,
    #             diff_recon_dec,
    #             diff_recon_emb_idxs,
    #             _,
    #             diff_recon_qidxs,
    #         ) = self.decode(enc, org_dec_h, use_ema=False)

    #         outputs.append(
    #             {
    #                 "org": self.make_dict(
    #                     org_enc, org_dec, org_emb_idxs, org_qidxs, org_spkr_cls
    #                 ),
    #                 "cv": self.make_dict(
    #                     cv_enc, cv_dec, cv_emb_idxs, cv_qidxs, cv_spkr_cls,
    #                 ),
    #                 "recon": self.make_dict(
    #                     recon_enc, recon_dec, recon_emb_idxs, recon_qidxs, None,
    #                 ),
    #                 "diffcv": self.make_dict(
    #                     cv_enc, diff_dec + x, cv_emb_idxs, cv_qidxs, diffcv_spkr_cls
    #                 ),
    #                 "diff_recon": self.make_dict(
    #                     diff_recon_enc,
    #                     diff_recon_dec,
    #                     diff_recon_emb_idxs,
    #                     diff_recon_qidxs,
    #                     None,
    #                 ),
    #             }
    #         )
    #         x = recon_dec.detach()
    #     return outputs

    def diff_decode(self, enc, dec_h, use_ema=True):
        dec = 0
        emb_idxs, emb_idx_qxs, qidxs = [], [], []
        for n in reversed(range(self.net_conf["n_vq_stacks"])):
            enc[n] = enc[n] + dec
            emb_idx, emb_idx_qx, qidx = self.quantizers[n](enc[n], use_ema=use_ema)
            emb_idxs.append(emb_idx)
            emb_idx_qxs.append(emb_idx_qx)
            qidxs.append(qidx)

            # decode
            if n != 0:
                dec = self.decoders[n](emb_idx_qx, c=None)
            else:
                dec = self.decoders[n](torch.cat(emb_idx_qxs, dim=1), c=dec_h)
                diff_dec = self.diff_decoder(torch.cat(emb_idx_qxs, dim=1), c=dec_h)
        return enc, dec, diff_dec, emb_idxs, emb_idx_qxs, qidxs

    def diff_decode_multi(self, enc, dec_h, use_ema=True):
        dec = 0
        emb_idxs, emb_idx_qxs, qidxs = [], [], []
        for n in reversed(range(self.net_conf["n_vq_stacks"])):
            enc[n] = enc[n] + dec
            emb_idx, emb_idx_qx, qidx = self.quantizers[n](enc[n], use_ema=use_ema)
            emb_idxs.append(emb_idx)
            emb_idx_qxs.append(emb_idx_qx)
            qidxs.append(qidx)

            # decode
            if n != 0:
                dec = self.diff_decoders[n](emb_idx_qx, c=None)
            else:
                dec = self.diff_decoders[n](torch.cat(emb_idx_qxs, dim=1), c=dec_h)
        return enc, dec, emb_idxs, emb_idx_qxs, qidxs

    def _construct_diff_decoder(self):
        nc = self.net_conf
        dec_in_channels = sum([nc["emb_dim"][i] for i in range(nc["n_vq_stacks"])])
        dec_out_channels = nc["output_size"]
        dec_aux_channels = nc["dec_aux_size"] + self.spkr_size
        self.diff_decoder = ParallelWaveGANGenerator(
            in_channels=dec_in_channels,
            out_channels=dec_out_channels,
            kernel_size=nc["kernel_size"][0],
            layers=nc["n_layers"][0] * nc["n_layers_stacks"][0],
            stacks=nc["n_layers_stacks"][0],
            residual_channels=nc["residual_channels"],
            gate_channels=128,
            skip_channels=64,
            aux_channels=dec_aux_channels,
            aux_context_window=0,
            dropout=0.0,
            bias=True,
            use_weight_norm=True,
            use_causal_conv=nc["causal"],
            upsample_conditional_features=False,
        )

    def _construct_diff_decoder_multi(self):
        nc = self.net_conf
        self.diff_decoders = nn.ModuleList()
        for n in range(nc["n_vq_stacks"]):
            if n == 0:
                dec_in_channels = sum(
                    [nc["emb_dim"][i] for i in range(nc["n_vq_stacks"])]
                )
                dec_out_channels = nc["output_size"]
                dec_aux_channels = nc["dec_aux_size"] + self.spkr_size
            elif n >= 1:
                dec_in_channels = nc["emb_dim"][n]
                dec_out_channels = nc["emb_dim"][n - 1]
                dec_aux_channels = 0
            self.diff_decoders.append(
                ParallelWaveGANGenerator(
                    in_channels=dec_in_channels,
                    out_channels=dec_out_channels,
                    kernel_size=nc["kernel_size"][n],
                    layers=nc["n_layers"][n] * nc["n_layers_stacks"][n],
                    stacks=nc["n_layers_stacks"][n],
                    residual_channels=nc["residual_channels"],
                    gate_channels=128,
                    skip_channels=64,
                    aux_channels=dec_aux_channels,
                    aux_context_window=0,
                    dropout=0.0,
                    bias=True,
                    use_weight_norm=True,
                    use_causal_conv=nc["causal"],
                    upsample_conditional_features=False,
                )
            )

    def remove_weight_norm(self):
        for n in range(self.net_conf["n_vq_stacks"]):
            self.encoders[n].remove_weight_norm()
            self.decoders[n].remove_weight_norm()
        self.diff_decoder.remove_weight_norm()
