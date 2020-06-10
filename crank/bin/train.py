#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2020 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Train VQ-VAE2 model

"""

import argparse
import logging
import random
import sys
import warnings

import joblib
import numpy as np
import torch

from pathlib import Path
from tensorboardX import SummaryWriter
from parallel_wavegan.models import ParallelWaveGANDiscriminator

from crank.net.module.vqvae2 import VQVAE2
from crank.net.module.diffvqvae2 import DiffVQVAE2
from crank.utils import load_yaml, open_scpdir, open_featsscp
from crank.net.trainer.utils import (
    get_optimizer,
    get_criterion,
    get_scheduler,
    get_dataloader,
)
from crank.net.trainer import (
    VQVAETrainer,
    LSGANTrainer,
    CycleVQVAETrainer,
    CycleGANTrainer,
    DiffVQVAETrainer,
)

warnings.simplefilter(action="ignore")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) " "%(levelname)s: %(message)s",
)

# Fix random variables
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def get_model(conf, spkr_size=0, device="cuda"):
    if conf["trainer_type"] in ["vqvae_diff"]:
        G = DiffVQVAE2(conf, spkr_size=spkr_size).to(device)
    else:
        G = VQVAE2(conf, spkr_size=spkr_size).to(device)

    # discriminator
    if conf["gan_type"] == "lsgan":
        output_channels = 1
    if conf["acgan_flag"]:
        output_channels += spkr_size

    if conf["discriminator_type"] == "pwg":
        D = ParallelWaveGANDiscriminator(
            in_channels=conf["input_size"],
            out_channels=output_channels,
            kernel_size=conf["kernel_size"][0],
            layers=conf["n_discriminator_layers"],
            conv_channels=64,
            dilation_factor=1,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.2},
            bias=True,
            use_weight_norm=True,
        )
    return {"G": G.to(device), "D": D.to(device)}


def load_checkpoint(model, checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    model["G"].load_state_dict(state_dict["model"]["G"])
    if "D" in state_dict["model"].keys():
        model["D"].load_state_dict(state_dict["model"]["D"])
    logging.info("load checkpoint: {}".format(checkpoint))
    return model, state_dict["steps"]


def main():
    # options for python
    description = "Train VQ-VAE model"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--flag",
        type=str,
        default="train",
        help='Flag for ["train", "eval", "reconstruction"]',
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="# of CPUs")
    parser.add_argument("--conf", type=str, help="ymal file for network parameters")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume model for re-training"
    )
    parser.add_argument("--scpdir", type=str, help="scp directory")
    parser.add_argument("--featdir", type=str, help="output feature directory")
    parser.add_argument(
        "--featsscp", type=str, help="specify feats.scp instead of using scp directory"
    )
    parser.add_argument("--expdir", type=str, help="exp directory")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert str(device) == "cuda", "ERROR: Do not accept CPU training."

    # load configure files
    conf = load_yaml(args.conf)
    for k, v in conf.items():
        logging.info("{}: {}".format(k, v))

    # load scp
    scp = {}
    featdir = Path(args.featdir) / conf["feature"]["label"]
    for phase in ["train", "dev", "eval"]:
        scp[phase] = open_scpdir(Path(args.scpdir) / phase)
        scp[phase]["feats"] = open_featsscp(featdir / phase / "feats.scp")
    if args.flag == "eval" and args.featsscp != "None":
        logging.info("Load feats.scp from {}".format(args.featsscp))
        scp[args.flag]["feats"] = open_featsscp(args.featsscp)

    expdir = Path(args.expdir) / Path(args.conf).stem
    expdir.mkdir(exist_ok=True, parents=True)
    spkr_size = len(scp["train"]["spkrs"])

    # load model
    model = get_model(conf, spkr_size, device)
    resume = 0
    if args.checkpoint != "None":
        model, resume = load_checkpoint(model, args.checkpoint)
    else:
        if args.flag in ["reconstruction", "eval"]:
            checkpoint = list(expdir.glob("*.pkl"))[-1]
            model, resume = load_checkpoint(model, checkpoint)

    # load others
    scaler = joblib.load(
        Path(args.expdir) / "{}_scaler.pkl".format(conf["feature"]["label"])
    )
    optimizer = get_optimizer(conf, model)
    criterion = get_criterion(conf)
    dataloader = get_dataloader(conf, scp, scaler, n_jobs=args.n_jobs, flag=args.flag)
    scheduler = get_scheduler(conf, optimizer)
    writer = {
        "train": SummaryWriter(logdir=args.expdir + "/runs/train-" + expdir.name),
        "dev": SummaryWriter(logdir=args.expdir + "/runs/dev-" + expdir.name),
    }

    ka = {
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "dataloader": dataloader,
        "writer": writer,
        "expdir": expdir,
        "conf": conf,
        "feat_conf": conf["feature"],
        "scheduler": scheduler,
        "device": device,
        "scaler": scaler,
        "resume": resume,
    }

    if conf["trainer_type"] == "vqvae":
        trainer = VQVAETrainer(**ka)
    elif conf["trainer_type"] == "lsgan":
        trainer = LSGANTrainer(**ka)
    elif conf["trainer_type"] == "cycle":
        trainer = CycleVQVAETrainer(**ka)
    elif conf["trainer_type"] == "cyclegan":
        trainer = CycleGANTrainer(**ka)
    elif conf["trainer_type"] == "vqvae_diff":
        trainer = DiffVQVAETrainer(**ka)
        assert conf["feat_type"] == "mcep", "feat_type must be mcep for DIFFVC"
    else:
        raise NotImplementedError(
            "conf['trainer_type']: {} is not supported.".format(conf["trainer_type"])
        )
    trainer.run(flag=args.flag)


if __name__ == "__main__":
    main()
