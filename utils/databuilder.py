import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from IPython.display import display
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class PatchedData(Dataset):
    def __init__(self):
        file = 'data.json'
        f = open(file)
        self.data = json.load(f)
        f.close()
        resh = 500
        resw = 500
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
                    gt[y1:y2, x1:x2] = 1
                print('File: {}; number: {}'.format(val['filename'], len(val['regions'])))
                gtimg = Image.fromarray(np.uint8(cm.gist_earth(gt.numpy()) * 255))
                self.imgsRaw.append(imgPIL)
                self.imgsPIL.append(Image.blend(imgPIL.convert("RGBA"), gtimg.convert("RGBA"), 0.5))
                self.imgs.append(image.permute(2, 0, 1))
                self.gts.append(gt)
                self.nums.append(len(val['regions']))
            except IOError:
                print('File not found: {}'.format(val['filename']))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        d, g = self.imgs[idx], self.gts[idx].unsqueeze(0)
        return d, g, idx


class BuildPatchedData:
    def __init__(self, patch_size=None):
        data = PatchedData()
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
            patched_gt = patchify(gt, (self.patch_width, self.patch_height), step=1)
            patched_image = patchify(image, (self.patch_width, self.patch_height, 3), step=1)

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

        num_samples = int(num_samples / 2)

        contains_node_index = np.random.choice(len(contains_node_image), num_samples)
        no_node_index = np.random.choice(len(no_node_image), num_samples)

        contains_node_image = contains_node_image[contains_node_index]
        no_node_image = no_node_image[no_node_index]

        X = np.vstack((contains_node_image, no_node_image))
        y = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))

        return X, y, patched_image_shape, patched_gt_shape

    def get_avg_patch_shape(self):
        with open("data.json") as f:
            metas = json.load(f)

        resizew, resizeh = 500, 500
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


class SegmentationData(Dataset):
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


class BuildSegmentationData:
    def __init__(self, patch_size=None):
        data = SegmentationData()
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


def visualize_patch(X_train, y_train, image_index):
    sample_x = X_train[image_index].reshape(64, 64, 3)
    sample_y = y_train[image_index].reshape(64, 64)
    f, axarr = plt.subplots(1, 2, figsize=(15, 15))
    axarr[0].imshow(sample_x)
    axarr[0].set_title("Image Patch")
    axarr[1].imshow(sample_y)
    axarr[1].set_title("Segmentation Patch")
    plt.tight_layout()


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
