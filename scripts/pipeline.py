import tomli
import shutil
import os
import argparse
from train import train
from sample import sample
from eval_catboost import train_catboost
from eval_mlp import train_mlp
from eval_simple import train_simple
import pandas as pd
import matplotlib.pyplot as plt
import lib
import torch


def load_config(path):
    with open(path, "rb") as f:
        return tomli.load(f)


def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        metavar="FILE",
        # default="exp/churn2/many-exps/config.toml",
        default="/mnt/SSD/github/biobank-anonymization-visualization/database/gan_optimize/test-churn2-tabddpm-lv_2-bs_256-epochs_5000-df_8-dm_7-dl_10-nl_3-lr_2.11e-04-model_mlp-moment_2-losscorcorr_7.61e+00-lossdis_9.39e-04-condvec_1/_config.toml",
        # default="exp/abalone/ddpm_mlp_best/config.toml",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        # default=False,
        default=True,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        # default=False,
        default=True,
    )
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--change_val", action="store_true", default=False)

    args = parser.parse_args()
    raw_config = lib.load_config(args.config)
    if "device" in raw_config:
        device = torch.device(raw_config["device"])
    else:
        device = torch.device("cuda:0")

    save_file(os.path.join(raw_config["parent_dir"], "config.toml"), args.config)

    if args.train:
        train(
            **raw_config["train"]["main"],
            **raw_config["diffusion_params"],
            parent_dir=raw_config["parent_dir"],
            real_data_path=raw_config["real_data_path"],
            model_type=raw_config["model_type"],
            model_params=raw_config["model_params"],
            T_dict=raw_config["train"]["T"],
            num_numerical_features=raw_config["num_numerical_features"],
            device=device,
            change_val=args.change_val,
        )
    if args.sample:
        sample(
            num_samples=raw_config["sample"]["num_samples"],
            batch_size=raw_config["sample"]["batch_size"],
            disbalance=raw_config["sample"].get("disbalance", None),
            **raw_config["diffusion_params"],
            parent_dir=raw_config["parent_dir"],
            real_data_path=raw_config["real_data_path"],
            model_path=os.path.join(raw_config["parent_dir"], "model.pt"),
            model_type=raw_config["model_type"],
            model_params=raw_config["model_params"],
            T_dict=raw_config["train"]["T"],
            num_numerical_features=raw_config["num_numerical_features"],
            device=device,
            seed=raw_config["sample"].get("seed", 0),
            change_val=args.change_val,
        )

    save_file(
        os.path.join(raw_config["parent_dir"], "info.json"),
        os.path.join(raw_config["real_data_path"], "info.json"),
    )
    if args.eval:
        if raw_config["eval"]["type"]["eval_model"] == "catboost":
            train_catboost(
                parent_dir=raw_config["parent_dir"],
                real_data_path=raw_config["real_data_path"],
                eval_type=raw_config["eval"]["type"]["eval_type"],
                T_dict=raw_config["eval"]["T"],
                seed=raw_config["seed"],
                change_val=args.change_val,
            )
        elif raw_config["eval"]["type"]["eval_model"] == "mlp":
            train_mlp(
                parent_dir=raw_config["parent_dir"],
                real_data_path=raw_config["real_data_path"],
                eval_type=raw_config["eval"]["type"]["eval_type"],
                T_dict=raw_config["eval"]["T"],
                seed=raw_config["seed"],
                change_val=args.change_val,
                device=device,
            )
        elif raw_config["eval"]["type"]["eval_model"] == "simple":
            train_simple(
                parent_dir=raw_config["parent_dir"],
                real_data_path=raw_config["real_data_path"],
                eval_type=raw_config["eval"]["type"]["eval_type"],
                T_dict=raw_config["eval"]["T"],
                seed=raw_config["seed"],
                change_val=args.change_val,
            )


if __name__ == "__main__":
    main()
