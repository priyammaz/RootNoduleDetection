import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from patchify import patchify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from google.colab import drive

drive.mount('/content/drive')

with open("drive/MyDrive/data.json") as f:
    metas = json.load(f)

resizew, resizeh = 500, 500
# Get Average Height and Width of Bounding Box
xdirection = []
ydirection = []
for key, val in metas.items():
    try:
        image = Image.open('drive/MyDrive/' + val['filename'])
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

buffer = 3
patch_width = int(np.ceil(np.mean(xdirection)) - buffer)
patch_height = int(np.ceil(np.mean(ydirection)) - buffer)

print("Patch Width: {} Patch Height: {}".format(patch_width, patch_height))


class MyData(Dataset):
    def __init__(self):
        file = "drive/MyDrive/data.json"
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
                imgPIL = Image.open('drive/MyDrive/' + val['filename'])
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


data = MyData()

## DO NOT RUN THIS AGAIN WITH THE IMAGE WRITING


patched_data = {"contains_node": {
    "image_patch": [],
    "gt_patch": []
},
    "empty_node": {
        "image_patch": [],
        "gt_patch": []
    }}

patched_data_images = {"contains_node": [],
                       "empty_node": []}

k = 0

for image, gt in zip(data.imgsRaw, data.gts):
    k = k + 1
    single_patches = []
    gt = np.array(gt)
    image = np.array(image)
    patched_gt = patchify(gt, (patch_width, patch_height), step=4).reshape(-1, patch_width, patch_height)
    patched_image = patchify(image, (patch_width, patch_height, 3), step=4).reshape(-1, patch_width, patch_height, 3)
    patches_image = patchify(image, (patch_width, patch_height, 3), step=4)
    bitMap = []

    for image_patch, gt_patch in zip(patched_image, patched_gt):
        # for i in range(patched_image.shape[0]):
        #   for j in range(patched_image.shape[1]):
        #     single_patch_img = image_patch[i, j, 0, :, :, :]
        # print(gt_patch.shape)

        if np.sum(gt_patch) > 0:
            # plt.imshow(image_patch)
            # plt.show()

            # plt.imshow(gt_patch)
            # plt.show()
            patched_data["contains_node"]["image_patch"].append(image_patch)
            patched_data["contains_node"]["gt_patch"].append(gt_patch)
            bitMap.append(1)

            # if image_patch is not None:
            #   if not cv2.imwrite('patches/images/positive/' + 'image_' + '_'+ str(image_patch.shape[0])+str(image_patch.shape[1])+'.png', image_patch):
            #     print("Could not write image")


        else:
            patched_data["empty_node"]["image_patch"].append(image_patch)
            patched_data["empty_node"]["gt_patch"].append(gt_patch)

            bitMap.append(0)

            # if single_patch_img is not None:
            #   if not cv2.imwrite('patches/images/negative/' + 'image_' + '_'+ str(i)+str(j)+'.png', single_patch_img):
            #     print("Could not write image")
    idx = 0
    for i in range(patches_image.shape[0]):
        for j in range(patches_image.shape[1]):
            single_patch_img = patches_image[i, j, 0, :, :, :]
            single_patches.append(single_patch_img)
            if bitMap[idx] > 0:
                cv2.imwrite(
                    'drive/MyDrive/patches/images/positive/' + 'image_' + '_' + str(k) + str(i) + str(j) + '.jpg',
                    single_patch_img)
                idx = idx + 1
            else:
                cv2.imwrite(
                    'drive/MyDrive/patches/images/negative/' + 'image_' + '_' + str(k) + str(i) + str(j) + '.jpg',
                    single_patch_img)
                idx = idx + 1

print("Total Patched with Node:", len(patched_data["contains_node"]["image_patch"]),
      len(patched_data["contains_node"]["gt_patch"]))
print("Total Patched without Node:", len(patched_data["empty_node"]["image_patch"]),
      len(patched_data["empty_node"]["gt_patch"]))

np.random.seed(42)
# PREP ARRAYS FOR TRAINING #
num_samples = 5000
contains_node_image = np.array([im.flatten() / 255 for im in patched_data["contains_node"]["image_patch"]])
no_node_image = np.array([im.flatten() / 255 for im in patched_data["empty_node"]["image_patch"]])

contains_node_index = np.random.choice(len(contains_node_image), num_samples)
no_node_index = np.random.choice(len(no_node_image), num_samples)

contains_node_image = contains_node_image[contains_node_index]
no_node_image = no_node_image[no_node_index]

X = np.vstack((contains_node_image, no_node_image))
y = np.vstack((np.ones((num_samples, 1)), np.zeros((num_samples, 1))))

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

plt.figure(figsize=(10, 10))
for label, group in label_groups:
    plt.scatter(group["c1"], group["c2"], marker="o", label=label, alpha=0.5)

plt.legend()
plt.title("PCA Clustering")
plt.show()

# TSNE Approach #
tnse = TSNE(n_components=2,
            random_state=42,
            learning_rate=200,
            n_jobs=-1)

tsne_data = tnse.fit_transform(X)
two_components = np.hstack((tsne_data, y.reshape(-1, 1)))
two_components = pd.DataFrame(data=two_components)
two_components.columns = ["c1", "c2", "label"]

label_groups = two_components.groupby("label")

plt.figure(figsize=(10, 10))
for label, group in label_groups:
    plt.scatter(group["c1"], group["c2"], marker="o", label=label, alpha=0.5)

plt.legend()
plt.title("TSNE Clustering")
plt.show()

# TSNE Approach #
tnse = TSNE(n_components=3,
            random_state=42,
            learning_rate=200,
            n_jobs=-1)

tsne_data = tnse.fit_transform(X)

fig = plt.figure(figsize=(20, 20))
ax = plt.axes(projection='3d')

sc = ax.scatter(tsne_data[:, 0].reshape(-1),
                tsne_data[:, 1].reshape(-1),
                tsne_data[:, 2].reshape(-1),
                c=y)

# PCA + Random Forest #
X_train, X_test, y_train, y_test = train_test_split(pca_output, y, test_size=0.1,
                                                    shuffle=True,
                                                    random_state=42)

X_train, X_test = X_train[:, :10], X_test[:, :10]

rfc = RandomForestClassifier(n_estimators=50,
                             min_samples_split=2,
                             min_samples_leaf=2,
                             max_features=5,
                             max_depth=10,
                             bootstrap=False,
                             random_state=0)

print("Fitting Data")
rfc.fit(X_train, y_train.squeeze())

print("Predicting Data")
y_train_pred = rfc.predict(X_train)
y_test_pred = rfc.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
plt.title("RFC Model Confusion Matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.show()

param_grid = {
    'n_estimators': [10, 30, 50, 75, 100, 150, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [6, 8, 10, 12, 14, 16],
    'criterion': ['gini', 'entropy']
}

X_train = pca_output[:, :10]

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
CV_rfc.fit(X_train, y.flatten())

print(CV_rfc.best_estimator_)
print(CV_rfc.best_score_)

print(X_train.shape)

from sklearn.svm import LinearSVC

svc = LinearSVC(loss='hinge', dual=True)
svc.fit(X_train, y_train)

print("Predicting Data")
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
plt.title("SVM Model Confusion Matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.show()

from sklearn.svm import SVC

model = SVC(kernel='poly', degree=2, gamma='auto', coef0=1, C=5)
model.fit(X_train, y_train)

print("Predicting Data")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
plt.title("SVM Model Confusion Matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.show()

model2 = SVC(kernel='rbf', gamma=0.5, C=0.1)
model2.fit(X_train, y_train)

print("Predicting Data")
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
plt.title("SVM Model Confusion Matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],
              'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'poly', 'linear']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
grid.fit(X_train, y.flatten())
best_params = grid.best_params_
print(f"Best params: {best_params}")
svm_clf = SVC(**best_params)
svm_clf.fit(X_train, y.flatten())

print(grid.best_estimator_)
print(grid.best_score_)

model2 = SVC(kernel='rbf', gamma=0.75, C=100)
model2.fit(X_train, y_train.squeeze())

print("Predicting Data")
y_train_pred = model2.predict(X_train)
y_test_pred = model2.predict(X_test)
print("Training Accuracy Score:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy Score:", accuracy_score(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 7))
plt.title("SVM Model Confusion Matrix")
sns.heatmap(cm, annot=True, cmap="YlGnBu")
plt.show()

"""#SVD + HOG approach"""

from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from PIL import Image
from numpy import *

pos_im_path = r"drive/MyDrive/patches/images/positive/"
neg_im_path = r"drive/MyDrive/patches/images/negative/"

pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing)
num_neg_samples = size(neg_im_listing)
print(num_pos_samples)
print(num_neg_samples)

orientations = 9
pixels_per_cell = (2, 2)
cells_per_block = (1, 1)
threshold = .03
data = []
labels = []

for file in pos_im_listing:
    img = Image.open(pos_im_path + '/' + file)
    gray = img.convert('L')
    fd = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)

print(len(data))
print(len(labels))

for file in neg_im_listing:
    img = Image.open(neg_im_path + '/' + file)
    # img = img.resize((64,128))
    gray = img.convert('L')
    # Now we calculate the HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(0)
# encode the labels, converting them from strings to integers
print(len(data))
print(len(labels))
le = LabelEncoder()
labels = le.fit_transform(labels)
# print(len(labels))

print(" Constructing training/testing split")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    np.array(data), labels, test_size=0.20, random_state=42)

print(" Training Linear SVM classifier")

modelSVC = LinearSVC()
modelSVC.fit(trainData, trainLabels)

print(" Evaluating classifier on test data")
predictions = modelSVC.predict(testData)
print(classification_report(testLabels, predictions))