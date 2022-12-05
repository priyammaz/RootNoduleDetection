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
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score
from sklearn.linear_model import RidgeClassifier

with open("data.json") as f:
    metas = json.load(f)

resizew, resizeh = 500, 500
# Get Average Height and Width of Bounding Box
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

buffer = 2
patch_width = int(np.ceil(np.mean(xdirection)) - buffer)
patch_height = int(np.ceil(np.mean(ydirection)) - buffer)

print("Patch Width: {} Patch Height: {}".format(patch_width, patch_height))


class MyData(Dataset):
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
        for key,val in self.data.items():
            try:
                imgPIL = Image.open(val['filename'])
                origwidth = imgPIL.size[0]
                origheight = imgPIL.size[1]
                imgPIL = imgPIL.resize((resw,resh),Image.LANCZOS)
                image = torch.Tensor(np.asarray(imgPIL)/255)
                gt = torch.zeros((image.shape[0:2]))
                for k2,v2 in val['regions'].items():
                    rectdata = v2['shape_attributes']
                    x1 = int(np.floor(resw*float(rectdata['x'])/origwidth))
                    y1 = int(np.floor(resh*float(rectdata['y'])/origheight))
                    x2 = int(np.ceil(resw*float(rectdata['x']+rectdata['width'])/origwidth))
                    y2 = int(np.ceil(resh*float(rectdata['y']+rectdata['height'])/origheight))
                    gt[y1:y2,x1:x2] = 1
                print('File: {}; number: {}'.format(val['filename'],len(val['regions'])))
                gtimg = Image.fromarray(np.uint8(cm.gist_earth(gt.numpy())*255))
                self.imgsRaw.append(imgPIL)
                self.imgsPIL.append(Image.blend(imgPIL.convert("RGBA"),gtimg.convert("RGBA"),0.5))
                self.imgs.append(image.permute(2,0,1))
                self.gts.append(gt)
                self.nums.append(len(val['regions']))
            except IOError:
                print('File not found: {}'.format(val['filename']))
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self,idx):
        d,g = self.imgs[idx], self.gts[idx].unsqueeze(0)
        return d, g, idx
data = MyData()

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
    patched_gt = patchify(gt, (patch_width, patch_height), step=4).reshape(-1, patch_width, patch_height)
    patched_image = patchify(image, (patch_width, patch_height, 3), step=4).reshape(-1, patch_width, patch_height, 3)
    for image_patch, gt_patch in zip(patched_image, patched_gt):
        if np.sum(gt_patch) > 0:
            patched_data["contains_node"]["image_patch"].append(image_patch)
            patched_data["contains_node"]["gt_patch"].append(gt_patch)


        else:
            patched_data["empty_node"]["image_patch"].append(image_patch)
            patched_data["empty_node"]["gt_patch"].append(gt_patch)


print("Total Patched with Node:", len(patched_data["contains_node"]["image_patch"]), len(patched_data["contains_node"]["gt_patch"]))
print("Total Patched without Node:", len(patched_data["empty_node"]["image_patch"]), len(patched_data["empty_node"]["gt_patch"]))

np.random.seed(42)
# PREP ARRAYS FOR TRAINING #
num_samples = 5000
contains_node_image = np.array([im.flatten()/255 for im in patched_data["contains_node"]["image_patch"]])
no_node_image = np.array([im.flatten()/255 for im in patched_data["empty_node"]["image_patch"]])

contains_node_index = np.random.choice(len(contains_node_image), num_samples)
no_node_index = np.random.choice(len(no_node_image), num_samples)

contains_node_image = contains_node_image[contains_node_index]
no_node_image = no_node_image[no_node_index]

X = np.vstack((contains_node_image, no_node_image))
y = np.vstack((np.ones((num_samples,1)), np.zeros((num_samples, 1))))


### PCA ###
pca = PCA(n_components=X.shape[-1])
pca_output = pca.fit_transform(X)

variance_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cumulative_variance_explained = np.cumsum(variance_explained)

plt.figure(figsize=(10, 10))

plt.plot(cumulative_variance_explained, linewidth=2)
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

two_components = pca_output[:, :2]
two_components = np.hstack((two_components, y.reshape(-1, 1)))
two_components = pd.DataFrame(data=two_components)
two_components.columns = ["c1", "c2", "label"]

label_groups = two_components.groupby("label")

plt.figure(figsize=(10,10))
for label, group in label_groups:
    plt.scatter(group["c1"], group["c2"], marker="o", label=label, alpha=0.5)

plt.legend()
plt.title("PCA Clustering")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True,
                                                    random_state=100)

### RIDGE REGRESSION ###
X_train, X_test = X_train[:, :10], X_test[:, :10]

ridge = RidgeClassifier(alpha=7,random_state=0)

print("Fitting Data")
ridge.fit(X_train, y_train.squeeze())

print("Predicting Data")
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

print("Training Precision Score:", precision_score(y_train, y_train_pred))
print("Testing Precision Score:", precision_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize = (10,7))
plt.title("Ridge Model Confusion Matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.show()

### PCA + RIDGE REGRESSION ###
X_train, X_test, y_train, y_test = train_test_split(pca_output, y, test_size=0.3,
                                                    shuffle=True,
                                                    random_state=100)

X_train, X_test = X_train[:, :10], X_test[:, :10]

ridge = RidgeClassifier(alpha=5,random_state=0)


print("Fitting Data")
ridge.fit(X_train, y_train.squeeze())

print("Predicting Data")
y_train_pred = ridge.predict(X_train)
y_test_pred = ridge.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

print("Training Precision Score:", precision_score(y_train, y_train_pred))
print("Testing Precision Score:", precision_score(y_test, y_test_pred))

cm_pca = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize = (10,7))
plt.title("Ridge Model Confusion Matrix")
sns.heatmap(cm_pca, annot=True, cmap="YlGnBu")
plt.show()






