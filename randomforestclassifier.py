import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


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

### BUILD DATASET ###
data = BuildData(patch_size=(3, 5))
X, y, patched_image_shape, patched_gt_shape = data.gen_patched_data()

### PCA OUTPUT ###
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

### PCA + RANDOM FOREST ###
# PCA + Random Forest #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    shuffle=True,
                                                    random_state=42)

X_train, X_test = X_train[:, :6], X_test[:, :6]

rfc = RandomForestClassifier(n_estimators=100,
                             min_samples_split=2,
                             min_samples_leaf=2,
                             max_features=6,
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

conf_mat = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize = (10,7))
plt.title("RFC Model Confusion Matrix")
sns.heatmap(conf_mat, annot=True, cmap="YlGnBu")
plt.show()

### RFC PARAMETER TUNING ###
param_grid = {
    'n_estimators': [10, 30, 50, 75, 100, 150, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [6, 8, 10, 12, 14, 16],
    'criterion' :['gini', 'entropy']
}

rfc = RandomForestClassifier()
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1)
CV_rfc.fit(X, y.flatten())

print(CV_rfc.best_estimator_)
print(CV_rfc.best_score_)


### INFERENCE ###
data = MyData()
images = [np.array(image) for image in data.imgsRaw]
gts = data.gts

sample_image, sample_gt = images[0], gts[0]

def predict(model, sample_image, sample_gt, pca=None, patch_shape=(3,5)):
    patched_image = patchify(sample_image, (patch_shape[0], patch_shape[1], 3), step=1)
    revert_shape_for_unpatch = patched_image.shape
    inference_data = patched_image.reshape(-1, patch_shape[0]*patch_shape[1]*3) / 255
    inference_shape = inference_data.shape
    if pca is not None:
        pca_output = pca[0].transform(inference_data)
        pca_output = pca_output[:, :pca[1]]
        predictions = model.predict(pca_output)
    else:
        predictions = model.predict(inference_data)
    output = np.ones(inference_shape)
    output = (output.T * predictions).T
    output = output.reshape(revert_shape_for_unpatch)
    output = unpatchify(output, imsize=(500,500,3))
    return output

output = predict(model=rfc, sample_image=sample_image, sample_gt=sample_gt, pca=(pca, 6), patch_shape=(2,2))
f, axarr = plt.subplots(1, 2, figsize=(15, 15))
axarr[0].imshow(sample_gt)
axarr[0].title.set_text('Ground Truth')
axarr[1].imshow(output[:, :, 0])
axarr[1].title.set_text('Predicted Node Locations')
plt.tight_layout()
plt.savefig("random_forest_prediction.png")