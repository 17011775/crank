#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Cyclic VQVAE trainer

"""


# from torch.nn.utils import clip_grad_norm
from crank.net.trainer.trainer_vqvae import VQVAETrainer


class CycleVQVAETrainer(VQVAETrainer):
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        dataloader,
        writer,
        expdir,
        conf,
        feat_conf,
        scheduler=None,
        scaler=None,
        resume=0,
        device="cuda",
        n_jobs=-1,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            dataloader,
            writer,
            expdir,
            conf,
            feat_conf,
            scheduler=scheduler,
            scaler=scaler,
            resume=resume,
            device=device,
            n_jobs=n_jobs,
        )
        self.cycle_flag = False
        self._check_cycle_start()

    def check_custom_start(self):
        self._check_cycle_start()

    def train(self, batch, phase="train"):
        loss = self._get_loss_dict()
        if self.cycle_flag:
            loss = self.forward_cycle(batch, loss, phase=phase)
        else:
            loss = self.forward_vqvae(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    def forward_cycle(self, batch, loss, phase="train"):
        enc_h = self._get_enc_h(batch)
        enc_h_cv = self._get_enc_h(batch, use_cvfeats=True)
        dec_h, spkrvec = self._get_dec_h(batch)
        dec_h_cv, spkrvec_cv = self._get_dec_h(batch, use_cvfeats=True)

        feats = batch["feats_sa"] if self.conf["spec_augment"] else batch["feats"]
        cycle_outputs = self.model["G"].cycle_forward(
            feats, enc_h, dec_h, enc_h_cv, dec_h_cv, spkrvec, spkrvec_cv
        )
        loss = self.calculate_vqvae_loss(batch, cycle_outputs[0]["org"], loss)
        loss = self.calculate_cyclevqvae_loss(batch, cycle_outputs, loss)

        if self.conf["speaker_adversarial"]:
            loss = self.calculate_spkradv_loss(
                batch, cycle_outputs[0]["org"], loss, phase=phase
            )

        loss["objective"] += loss["G"]
        if phase == "train":
            self.optimizer["G"].zero_grad()
            loss["G"].backward()
            # clip_grad_norm(self.model["G"].parameters(), self.conf["clip_grad_norm"])
            self.optimizer["G"].step()
        return loss

    def _parse_cyclevqvae_loss(self, loss):
        for c in range(self.conf["n_cycles"]):
            alpha_cycle = self.conf["alphas"]["cycle"] ** (c + 1)
            # for cv
            lbl = "{}cyc_{}".format(c, "cv")

            if self.conf["encoder_spkr_classifier"]:
                loss["G"] += (
                    alpha_cycle * self.conf["alphas"]["ce"] * loss["ce" + "_" + lbl]
                )

            # for recon
            lbl = "{}cyc_{}".format(c, "recon")
            for k in ["l1", "mse", "stft"]:
                loss["G"] += alpha_cycle * self.conf["alphas"][k] * loss[k + "_" + lbl]
            for n in range(self.conf["n_vq_stacks"]):
                loss["G"] += (
                    alpha_cycle
                    * self.conf["alphas"]["commit"][n]
                    * loss["{}{}_{}".format("commit", n, lbl)]
                )
                if not self.conf["ema_flag"]:
                    loss["G"] += (
                        alpha_cycle
                        * self.conf["alphas"]["dict"][n]
                        * loss["{}{}_{}".format("dict", n, lbl)]
                    )
        return loss

    def calculate_cyclevqvae_loss(self, batch, outputs, loss):
        mask = batch["mask"]
        for c in range(self.conf["n_cycles"]):
            for io in ["cv", "recon"]:
                lbl = "{}cyc_{}".format(c, io)
                o = outputs[c][io]
                if io == "cv":
                    if self.conf["encoder_spkr_classifier"]:
                        loss["ce_{}".format(lbl)] = self.criterion["ce"](
                            o["spkr_cls"].reshape(-1, o["spkr_cls"].size(2)),
                            batch["cv_h_scalar"].reshape(-1),
                        )
                elif io == "recon":
                    feats = batch["feats"]
                    decoded = o["decoded"]
                    loss["l1_{}".format(lbl)] = self.criterion["fl1"](
                        decoded, feats, mask=mask
                    )
                    loss["mse_{}".format(lbl)] = self.criterion["fmse"](
                        decoded, feats, mask=mask
                    )
                    loss["stft_{}".format(lbl)] = self.criterion["fstft"](
                        o["decoded"], batch["feats"]
                    )
                    for n in range(self.conf["n_vq_stacks"]):
                        loss["commit{}_{}".format(n, lbl)] = self.criterion["mse"](
                            o["encoded"][n].masked_select(mask),
                            o["emb_idx"][n].masked_select(mask).detach(),
                        )
                        if not self.conf["ema_flag"]:
                            loss["dict{}_{}".format(n, lbl)] = self.criterion["mse"](
                                o["emb_idx"][n].masked_select(mask),
                                o["encoded"][n].masked_select(mask).detach(),
                            )
        loss = self._parse_cyclevqvae_loss(loss)
        return loss

    def _check_cycle_start(self):
        if self.steps > self.conf["n_steps_cycle_start"]:
            self.cycle_flag = True
