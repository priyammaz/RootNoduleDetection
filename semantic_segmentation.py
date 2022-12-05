import numpy as np
import json
from PIL import Image
from matplotlib import cm
from patchify import patchify
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2


class MyData(Dataset):
    def __init__(self):
        file = 'data.json'
        f = open(file)
        self.data = json.load(f)
        f.close()
        resh = 2000
        resw = 2000
        self.imgs = []
        self.gts = []
        self.nums = []
        self.imgsPIL = []
        self.imgsRaw = []
        for key, val in self.data.items():
            try:
                imgPIL = Image.open(val['filename'])
                origwidth = imgPIL.size[0]
                origheight = imgPIL.size[1]
                imgPIL = imgPIL.resize((resw, resh), Image.LANCZOS)
                image = torch.Tensor(np.asarray(imgPIL) / 255)
                gt = torch.zeros((image.shape[0:2]))
                for k2, v2 in val['regions'].items():
                    rectdata = v2['shape_attributes']
                    x1 = int(np.floor(resw * float(rectdata['x']) / origwidth))
                    y1 = int(np.floor(resh * float(rectdata['y']) / origheight))
                    x2 = int(np.ceil(resw * float(rectdata['x'] + rectdata['width']) / origwidth))
                    y2 = int(np.ceil(resh * float(rectdata['y'] + rectdata['height']) / origheight))
                    radius = ((x2 - x1) / 2 + (y2 - y1) / 2) / 2
                    mask = self.create_circular_mask(resh, resw, center=((x2 + x1) / 2, (y2 + y1) / 2), radius=radius)
                    gt[mask] = 1

                print('File: {}; number: {}'.format(val['filename'], len(val['regions'])))
                gtimg = Image.fromarray(np.uint8(cm.gist_earth(gt.numpy()) * 255))
                self.imgsRaw.append(imgPIL)
                self.imgsPIL.append(Image.blend(imgPIL.convert("RGBA"), gtimg.convert("RGBA"), 0.5))
                self.imgs.append(image.permute(2, 0, 1))
                gt[gt > 0] = 1
                self.gts.append(gt)
                self.nums.append(len(val['regions']))
            except IOError:
                print('File not found: {}'.format(val['filename']))

    def create_circular_mask(self, h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        d, g = self.imgs[idx], self.gts[idx].unsqueeze(0)
        return d, g, idx


class BuildData:
    def __init__(self, patch_size=None):
        data = MyData()
        self.imgsRaw, self.gts = data.imgsRaw, data.gts

        if patch_size is None:
            self.patch_width, self.patch_height = self.get_avg_patch_shape()
        else:
            self.patch_width, self.patch_height = patch_size

        print(f"Using Patch Size ({self.patch_width}, {self.patch_height})")

    def gen_patched_data(self, num_samples=50000):
        patched_data = {"contains_node": {
            "image_patch": [],
            "gt_patch": []
        },
            "empty_node": {
                "image_patch": [],
                "gt_patch": []
            }}

        for image, gt in zip(data.imgsRaw, data.gts):
            gt = np.array(gt)
            image = np.array(image)
            patched_gt = patchify(gt, (self.patch_width, self.patch_height), step=32)
            patched_image = patchify(image, (self.patch_width, self.patch_height, 3), step=32)
            patched_gt_shape = patched_gt.shape
            patched_image_shape = patched_image.shape

            patched_gt = patched_gt.reshape(-1, self.patch_width, self.patch_height)
            patched_image = patched_image.reshape(-1, self.patch_width, self.patch_height, 3)

            for image_patch, gt_patch in zip(patched_image, patched_gt):
                if np.sum(gt_patch) > 0:
                    patched_data["contains_node"]["image_patch"].append(image_patch)
                    patched_data["contains_node"]["gt_patch"].append(gt_patch)


                else:
                    patched_data["empty_node"]["image_patch"].append(image_patch)
                    patched_data["empty_node"]["gt_patch"].append(gt_patch)

        print("Total Patches with Node:", len(patched_data["contains_node"]["image_patch"]))
        print("Total Patches without Node:", len(patched_data["empty_node"]["image_patch"]))

        contains_node_image = np.array([im.flatten() / 255 for im in patched_data["contains_node"]["image_patch"]])
        no_node_image = np.array([im.flatten() / 255 for im in patched_data["empty_node"]["image_patch"]])

        contains_node_seg = np.array([im.flatten() / 255 for im in patched_data["contains_node"]["gt_patch"]])
        no_node_seg = np.array([im.flatten() / 255 for im in patched_data["empty_node"]["gt_patch"]])

        num_samples = int(num_samples / 2)

        contains_node_index = np.random.choice(len(contains_node_image), num_samples)
        no_node_index = np.random.choice(len(no_node_image), num_samples)

        contains_node_image = contains_node_image[contains_node_index]
        no_node_image = no_node_image[no_node_index]

        contains_node_seg = contains_node_seg[contains_node_index]
        no_node_seg = no_node_seg[no_node_index]

        X = np.vstack((contains_node_image, no_node_image))
        y = np.vstack((contains_node_seg, no_node_seg))

        return X, y, patched_image_shape, patched_gt_shape

    def get_avg_patch_shape(self):
        with open("data.json") as f:
            metas = json.load(f)

        resizew, resizeh = 2000, 2000
        xdirection = []
        ydirection = []

        for key, val in metas.items():
            try:
                image = Image.open(val["filename"])
                h, w = image.size
                image = image.resize((resizew, resizeh), Image.LANCZOS)
                gt = torch.zeros((resizew, resizeh))
                for key2, val2 in val["regions"].items():
                    rectdata = val2["shape_attributes"]
                    x1 = int(np.floor(resizew * float(rectdata['x']) / w))
                    y1 = int(np.floor(resizeh * float(rectdata['y']) / h))
                    x2 = int(np.ceil(resizew * float(rectdata['x'] + rectdata['width']) / w))
                    y2 = int(np.ceil(resizeh * float(rectdata['y'] + rectdata['height']) / h))

                    xdirection.append(abs(x2 - x1))
                    ydirection.append(abs(y2 - y1))
            except:
                pass

        patch_width = int(np.ceil(np.mean(xdirection)))
        patch_height = int(np.ceil(np.mean(ydirection)))

        return patch_width, patch_height


data = BuildData(patch_size=(64, 64))
X, y, patched_image_shape, patched_gt_shape = data.gen_patched_data(num_samples=5800)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
y_train[y_train > 0] = 1
y_test[y_test > 0] = 1

### DEFINE MODEL ###
### UNET MODEL CODE BORROWED FROM https://github.com/milesial/Pytorch-UNet ###
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SegLoader(Dataset):
    def __init__(self, dataset, transforms=None):
        self.X, self.y = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = np.moveaxis(self.X[idx].reshape(64, 64, 3), -1, 0)
        sample_y = self.y[idx].reshape(64, 64)
        if np.random.random() > 0.5:
            sample_X = np.flip(sample_X, 1)
            sample_y = np.flip(sample_y, 0)

        if np.random.random() > 0.5:
            sample_X = np.flip(sample_X, 2)
            sample_y = np.flip(sample_y, 1)

        sample_X = torch.Tensor(sample_X.copy())
        sample_y = torch.Tensor(sample_y.copy())

        return sample_X, sample_y


class SegGrayLoader(Dataset):
    def __init__(self, dataset, transforms=None):
        self.X, self.y = dataset

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_X = self.X[idx].reshape(64, 64, 3).astype(np.float32)
        sample_X = cv2.cvtColor(sample_X, cv2.COLOR_RGB2GRAY)
        sample_y = self.y[idx].reshape(64, 64)
        if np.random.random() > 0.5:
            sample_X = np.flip(sample_X, 0)
            sample_y = np.flip(sample_y, 0)

        if np.random.random() > 0.5:
            sample_X = np.flip(sample_X, 1)
            sample_y = np.flip(sample_y, 1)

        sample_X = torch.Tensor(sample_X.copy()).unsqueeze(0)
        sample_y = torch.Tensor(sample_y.copy())

        return sample_X, sample_y


train_dataset = SegGrayLoader(dataset=(X_train, y_train), transforms=None)
test_dataset = SegGrayLoader(dataset=(X_test, y_test), transforms=None)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

### TRAIN MODEL ###
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(1, 1)
model = model.to(device)

EPOCHS = 15
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(params=model.parameters())

agg_training_loss = []
agg_validation_loss = []
for epoch in range(1, EPOCHS + 1):
    training_losses = []
    validation_losses = []
    ### TRAINING LOOP ###
    model.train()
    for X, y in tqdm(trainloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model.forward(X).squeeze()
        loss = loss_fn(out, y)
        training_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    ### TRAINING LOOP ###
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(trainloader):
            X, y = X.to(device), y.to(device)
            out = model.forward(X).squeeze()
            loss = loss_fn(out, y)
            validation_losses.append(loss.item())

    agg_training = np.mean(training_losses)
    agg_validation = np.mean(validation_losses)

    print(f"Epoch: {epoch}, Training Loss: {agg_training}, Validation Loss: {agg_validation}")
    agg_training_loss.append(agg_training)
    agg_validation_loss.append(agg_validation)