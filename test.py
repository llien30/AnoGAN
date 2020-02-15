from weights import G, D

from libs.dataloader import Dataset
from libs.transform import ImageTransform
from libs.loss import Anomaly_Score

from torchvision.utils import save_image
from addict import Dict
import numpy as np
from PIL import Image
import pandas as pd

import wandb

G_update = G
D_update = D

def get_parameters():
    '''
    make parser to get parameters
    '''

    parser = argparse.ArgumentParser(
        description='take parameters like the path of csv file ...')

    parser.add_argument('config', type=str, help='path of a config file for the test')

    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Add --no_wandb option'
    )

    return parser.parse_args()

args = get_parameters()

CONFIG = Dict(yaml.safe_load(open(args.config)))

if not args.no_wandb:
    wandb.init(
        config = CONFIG,
        name = CONFIG.name,
        project = 'mnist',  # have to change when you want to change project
        job_type = 'anomaly detection'
    )

test_dataset = Dataset(
    csv_file=CONFIG.test_csv_file, transform=ImageTransform(mean, std))

test_dataloader = DataLoader(
    test_dataset, batch_size=CONFIG.test_batch_size, shuffle=True)

x = []
for sample in test_dataloader:
    img = sample[img]
    x.append(img)

x = x.to(device)

test_z = torch.randn(CONFIG.test_batch_size, CONFIG.z_dim, 1, 1).to(device)
test_z.requires_grad = True
z_optimizer = torch.optim.Adam([test_z], lr=1e-3)

for epoch in range(5000+1):
    fake_image = G_update(test_z)
    loss, _, _ = Anomaly_Score(x, fake_image, D_update, Lambda=0.1)

    z_optimizer.zero_grad()
    loss.backward()
    z_optimizer.step()

    if epoch%1000 == 0:
        print('epoch {} || Loss total : {:.0f}'.format(epoch, loss.item()))

fake_img = G_update(test_z)

loss, loss_each, residual_loss_each = Anomaly_Score(x, fake_img, D_update, Lambda=0.1)

loss_each = loss_each.cpu().detach().numpy()

loss_each = np.round(loss_each, 0)
img_no = np.array([i for i in range(CONFIG.test_batch_size)])

save_image(x, 'test_image.png')
save_image(fake_img, 'test_fakeimage.png')

df = pd.DataFrame({
    'No.':img_no,
    'loss':loss_each},
    columns=['No.','loss'])

if not os.path.exists(CONFIG.test_save_dir):
    os.makedirs(CONFIG.test_save_dir)

df.to_csv(os.path.join(CONFIG.test_save_dir,'{}.csv').format(split), index=None)

if not no_wandb:
    wandb.log({'Anomaly_score': loss_each})

    test_img = Image.open('test_image.png')
    test_fakeimg = Image.open('test_fakeimage.png')
    wandb.log({
        "test_image": [wandb.Image(test_img)],
        "test_fakeimage": [wandb.Image(test_fakeimg)]})

