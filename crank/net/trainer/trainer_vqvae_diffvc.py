#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 K. Kobayashi <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Differential cyclic VQVAE

"""


import numpy as np
from joblib import Parallel, delayed

from crank.net.trainer.trainer_vqvae import VQVAETrainer
from crank.utils import to_numpy, world2wav, diff2wav


class DiffVQVAETrainer(VQVAETrainer):
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
        self.diff_flag = False
        self._check_diff_start()

    def check_custom_start(self):
        self._check_diff_start()

    def train(self, batch, phase="train"):
        loss = self._get_loss_dict()
        if self.diff_flag:
            loss = self.forward_diff(batch, loss, phase=phase)
        else:
            loss = self.forward_vqvae(batch, loss, phase=phase)
        loss_values = self._parse_loss(loss)
        self._flush_writer(loss, phase)
        return loss_values

    def forward_diff(self, batch, loss, phase="train"):
        h = self._generate_conditions(batch)
        h_cv = self._generate_conditions(batch, use_cvfeats=True)

        outputs = self.model["G"].diff_forward(
            batch["feats"], org_dec_h=h, cv_dec_h=h_cv
        )
        loss = self.calculate_vqvae_loss(batch, outputs["org"], loss)
        loss = self.calculate_diff_vqvae_loss(batch, outputs, loss)

        # if cycle
        # loss = self.calculate_vqvae_loss(batch, outputs[0]["org"], loss)
        # loss = self.calculate_cycle_diff_vqvae_loss(batch, outputs, loss)

        loss["objective"] += loss["generator"]
        if phase == "train":
            self.optimizer["generator"].zero_grad()
            loss["generator"].backward()
            self.optimizer["generator"].step()
        return loss

    def calculate_diff_vqvae_loss(self, batch, outputs, loss):
        mask = batch["mask"]
        loss = self._calculate_feature_matching_loss(
            outputs["cv"]["decoded"].detach(),
            outputs["diffcv"]["decoded"],
            mask,
            loss,
            "match",
        )
        for k in ["l1", "mse", "stft"]:
            loss["generator"] += self.conf["alphas"][k] * loss[k + "_match"]

        for io in ["cv", "diffcv"]:
            o = outputs[io]
            for n in range(self.conf["n_vq_stacks"]):
                loss["commit{}_{}".format(n, io)] = self.criterion["mse"](
                    o["encoded"][n].masked_select(mask),
                    o["emb_idx"][n].masked_select(mask).detach(),
                )
                loss["generator"] += (
                    self.conf["alphas"]["commit"][n] * loss["commit{}_{}".format(n, io)]
                )

        return loss

    def _calculate_feature_matching_loss(self, feats, decoded, mask, loss, lbl):
        masked_feats = feats.masked_select(mask)
        masked_decoded = decoded.masked_select(mask)
        loss["l1_{}".format(lbl)] = self.criterion["l1"](masked_feats, masked_decoded)
        loss["mse_{}".format(lbl)] = self.criterion["mse"](masked_feats, masked_decoded)
        loss["stft_{}".format(lbl)] = self.criterion["stft"](feats, decoded)
        return loss

    def calculate_cycle_diff_vqvae_loss(self, batch, outputs, loss):
        mask = batch["mask"]
        for c in range(self.conf["n_cycles"]):
            for io in ["cv", "recon", "diffcv", "diff_recon"]:
                lbl = "{}cyc_{}".format(c, io)
                o = outputs[c][io]
                if io in ["cv", "diffcv"]:
                    loss["ce_{}".format(lbl)] = self.criterion["ce"](
                        o["spkr_cls"].reshape(-1, o["spkr_cls"].size(2)),
                        batch["cv_h_scalar"].reshape(-1),
                    )
                elif io in ["recon", "diff_recon"]:
                    loss = self._calculate_feature_loss(
                        batch["feats"], o["decoded"], mask, loss, lbl
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
            # feature matching loss
            lbl = "{}cyc_{}".format(c, "match")
            loss = self._calculate_feature_loss(
                outputs[c]["cv"]["decoded"],
                outputs[c]["diffcv"]["decoded"],
                mask,
                loss,
                lbl,
            )
        loss = self._parse_cycle_diff_vqvae_loss(loss)
        return loss

    def _parse_cycle_diff_vqvae_loss(self, loss):
        for c in range(self.conf["n_cycles"]):
            alpha_cycle = self.conf["alphas"]["cycle"] ** (c + 1)
            # for cv
            for io in ["cv", "diffcv"]:
                lbl = "{}cyc_{}".format(c, io)
                loss["generator"] += (
                    alpha_cycle * self.conf["alphas"]["ce"] * loss["ce" + "_" + lbl]
                )

            # for recon
            for io in ["recon", "diff_recon"]:
                lbl = "{}cyc_{}".format(c, io)
                for k in ["l1", "mse", "stft"]:
                    loss["generator"] += (
                        alpha_cycle * self.conf["alphas"][k] * loss[k + "_" + lbl]
                    )
                for n in range(self.conf["n_vq_stacks"]):
                    loss["generator"] += (
                        alpha_cycle
                        * self.conf["alphas"]["commit"][n]
                        * loss["{}{}_{}".format("commit", n, lbl)]
                    )
                    if not self.conf["ema_flag"]:
                        loss["generator"] += (
                            alpha_cycle
                            * self.conf["alphas"]["dict"][n]
                            * loss["{}{}_{}".format("dict", n, lbl)]
                        )

            # for cv matching
            lbl = "{}cyc_{}".format(c, "match")
            for k in ["l1", "mse", "stft"]:
                loss["generator"] += (
                    alpha_cycle * self.conf["alphas"][k] * loss[k + "_" + lbl]
                )

        return loss

    def _generate_cvwav(self, batch, outputs, cv_spkr_name=None, tdir="dev_wav"):
        tdir = self.expdir / tdir / str(self.steps)
        feats = self._store_features(batch, outputs, cv_spkr_name, tdir)
        feats = self._store_diff_feature(feats, batch, outputs, cv_spkr_name, tdir)

        # generate wav
        self._save_decoded_world(feats)
        self._save_decoded_world_diffvc(feats)

    def _store_diff_feature(self, feats, batch, outputs, cv_spkr_name, tdir):
        feat_type = self.conf["feat_type"]
        decoded_diff = outputs["decoded_diff"]
        for n in range(decoded_diff.size(0)):
            org_spkr_name = batch["org_spkr_name"][n]
            cv_spkr_name = org_spkr_name if cv_spkr_name is None else cv_spkr_name
            wavf = tdir / "{}_org-{}_cv-{}.wav".format(
                batch["flbl"][n], org_spkr_name, cv_spkr_name
            )

            flen = batch["flen"][n]
            normed_orgfeat = to_numpy(batch["feats"][n][:flen])
            org_feat = self.scaler[feat_type].inverse_transform(normed_orgfeat)
            feat = to_numpy(decoded_diff[n][:flen])
            feats[wavf]["feat_diffvc"] = self.scaler[feat_type].inverse_transform(feat)
            feats[wavf]["feat_diffvc"][:, 0] = org_feat[:, 0]
            feats[wavf]["feat_diff"] = feats[wavf]["feat_diffvc"] - org_feat
            feats[wavf]["org_feat"] = org_feat
        return feats

    def _save_decoded_world_diffvc(self, feats):
        Parallel(n_jobs=self.n_jobs)(
            [
                delayed(world2wav)(
                    v["f0"][:, 0].astype(np.float64),
                    v["feat_diffvc"].astype(np.float64),
                    v["cap"].astype(np.float64),
                    wavf=str(k) + ".diff_vocoder.wav",
                    fs=self.conf["feature"]["fs"],
                    fftl=self.conf["feature"]["fftl"],
                    shiftms=self.conf["feature"]["shiftms"],
                    alpha=self.conf["feature"]["mcep_alpha"],
                )
                for k, v in feats.items()
            ]
        )
        # Parallel(n_jobs=self.n_jobs)(
        #     [
        #         delayed(diff2wav)(
        #             v["org_wav"].astype(np.float64),
        #             v["feat_diff"].astype(np.float64),
        #             v["org_feat"].astype(np.float64),
        #             wavf=str(k) + "diff.wav",
        #             fs=self.conf["feature"]["fs"],
        #             fftl=self.conf["feature"]["fftl"],
        #             shiftms=self.conf["feature"]["shiftms"],
        #             alpha=self.conf["feature"]["mcep_alpha"],
        #         )
        #         for k, v in feats.items()
        #     ]
        # )

    def _check_diff_start(self):
        if self.steps > self.conf["n_steps_diff_start"]:
            self.diff_flag = True
