from model import NetD, NetG

from libs.dataloader import Dataset
from libs.transform import ImageTransform
from libs.loss import Anomaly_Score

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import os
import argparse
import yaml
from addict import Dict
import numpy as np
from PIL import Image
import pandas as pd

import wandb


def get_parameters():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(
        description="take parameters like the path of csv file ..."
    )

    parser.add_argument("config", type=str, help="path of a config file for the test")

    parser.add_argument("--no_wandb", action="store_true", help="Add --no_wandb option")

    return parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_parameters()

CONFIG = Dict(yaml.safe_load(open(args.config)))

if not args.no_wandb:
    wandb.init(
        config=CONFIG,
        name=CONFIG.name,
        project="mnist",  # have to change when you want to change project
        job_type="anomaly detection",
    )

state_dict_G = torch.load("weights/G.prm", map_location=lambda storage, loc: storage)
state_dict_D = torch.load("weights/D.prm", map_location=lambda storage, loc: storage)

G_update = NetG(CONFIG)
D_update = NetD(CONFIG)

G_update.load_state_dict(state_dict_G)
D_update.load_state_dict(state_dict_D)

G_update.to(device)
D_update.to(device)

mean = (0.5,)
std = (0.5,)

test_dataset = Dataset(
    csv_file=CONFIG.test_csv_file, transform=ImageTransform(mean, std)
)

test_dataloader = DataLoader(
    test_dataset, batch_size=CONFIG.test_batch_size, shuffle=True
)

for sample in test_dataloader:
    test_img = sample["img"]

# print(test_img)
test_img = test_img.to(device)

test_z = torch.randn(CONFIG.test_batch_size, CONFIG.z_dim, 1, 1).to(device)
test_z.requires_grad = True
z_optimizer = torch.optim.Adam([test_z], lr=1e-3)

for epoch in range(5000 + 1):
    fake_image = G_update(test_z)
    loss, _, _ = Anomaly_Score(test_img, fake_image, D_update, Lambda=0.1)

    z_optimizer.zero_grad()
    loss.backward()
    z_optimizer.step()

    if epoch % 1000 == 0:
        print("epoch {} || Loss total : {:.0f}".format(epoch, loss.item()))

fake_img = G_update(test_z)

loss, loss_each, residual_loss_each = Anomaly_Score(
    test_img, fake_img, D_update, Lambda=0.1
)

loss_each = loss_each.cpu().detach().numpy()

loss_each = np.round(loss_each, 0)
img_no = np.array([i for i in range(CONFIG.test_batch_size)])

save_image(test_img, "test_image.png")
save_image(fake_img, "test_fakeimage.png")

df = pd.DataFrame({"No.": img_no, "loss": loss_each}, columns=["No.", "loss"])

if not os.path.exists(CONFIG.test_save_dir):
    os.makedirs(CONFIG.test_save_dir)

df.to_csv(os.path.join(CONFIG.test_save_dir, "{}.csv").format("result"), index=None)

if not args.no_wandb:
    wandb.log({"Anomaly_score": loss_each})

    test_img = Image.open("test_image.png")
    test_fakeimg = Image.open("test_fakeimage.png")
    wandb.log(
        {
            "test_image": [wandb.Image(test_img)],
            "test_fakeimage": [wandb.Image(test_fakeimg)],
        }
    )

