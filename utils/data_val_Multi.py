import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import cv2
import random
from PIL import ImageEnhance
import torch


# ==============================
# ==============================
def cv_random_flip_pair(rgb_img, t_img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        rgb_img = rgb_img.transpose(Image.FLIP_LEFT_RIGHT)
        t_img = t_img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return rgb_img, t_img, label, edge


def randomCrop_pair(rgb_img, t_img, label, edge):
    border = 30
    image_width, image_height = rgb_img.size
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)

    left = (image_width - crop_win_width) >> 1
    top = (image_height - crop_win_height) >> 1
    right = (image_width + crop_win_width) >> 1
    bottom = (image_height + crop_win_height) >> 1
    region = (left, top, right, bottom)

    rgb_img = rgb_img.crop(region)
    t_img = t_img.crop(region)
    label = label.crop(region)
    edge = edge.crop(region)
    return rgb_img, t_img, label, edge


def randomRotation_pair(rgb_img, t_img, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        rgb_img = rgb_img.rotate(random_angle, mode)
        t_img = t_img.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return rgb_img, t_img, label, edge


# ==============================
# ==============================
def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) == 0:
            img[randX, randY] = 0
        else:
            img[randX, randY] = 255
    return Image.fromarray(img)


class PolypDataset_Multi(data.Dataset):
    def __init__(self, rgb_root, thermal_root, gt_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.rgb_images = sorted(
            [os.path.join(rgb_root, f) for f in os.listdir(rgb_root) if f.lower().endswith(('.jpg', '.png'))]
        )
        self.thermal_images = sorted(
            [os.path.join(thermal_root, f) for f in os.listdir(thermal_root) if f.lower().endswith(('.jpg', '.png'))]
        )
        self.gts = sorted(
            [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.tif'))]
        )
        self.edges = sorted(
            [os.path.join(edge_root, f) for f in os.listdir(edge_root) if f.lower().endswith(('.jpg', '.png'))]
        )

        self.filter_files()
        self.size = len(self.rgb_images)

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()
        ])

        self.kernel = np.ones((3, 3), np.uint8)

    def __getitem__(self, index):
        rgb_img = self.rgb_loader(self.rgb_images[index])
        thermal_img = self.rgb_loader(self.thermal_images[index])
        gt = self.binary_loader(self.gts[index])

        edge = cv2.imread(self.edges[index], cv2.IMREAD_GRAYSCALE)
        edge = cv2.dilate(edge, self.kernel, iterations=1)
        edge = Image.fromarray(edge)

        rgb_img, thermal_img, gt, edge = cv_random_flip_pair(rgb_img, thermal_img, gt, edge)
        rgb_img, thermal_img, gt, edge = randomCrop_pair(rgb_img, thermal_img, gt, edge)
        rgb_img, thermal_img, gt, edge = randomRotation_pair(rgb_img, thermal_img, gt, edge)

        rgb_img = colorEnhance(rgb_img)
        thermal_img = colorEnhance(thermal_img)
        gt = randomPeper(gt)
        edge = randomPeper(edge)

        rgb_img = self.rgb_transform(rgb_img)
        thermal_img = self.thermal_transform(thermal_img)
        gt = self.gt_transform(gt)
        edge = self.edge_transform(edge)
        edge_small = self.Threshold_process(edge)

        return rgb_img, thermal_img, gt, edge_small

    def filter_files(self):
        orig_rgb = self.rgb_images.copy()
        orig_thermal = self.thermal_images.copy()
        orig_gts = self.gts.copy()
        orig_edges = self.edges.copy()
        assert len(orig_rgb) == len(orig_thermal) == len(orig_gts) == len(orig_edges), "初始文件数量不一致！"

        rgb_list, thermal_list, gt_list, edge_list = [], [], [], []
        for rgb_path, thermal_path, gt_path, edge_path in zip(orig_rgb, orig_thermal, orig_gts, orig_edges):
            try:
                rgb_img = Image.open(rgb_path)
                thermal_img = Image.open(thermal_path)
                gt_img = Image.open(gt_path)
                edge_img = Image.open(edge_path)
            except Exception:
                continue
            if rgb_img.size == thermal_img.size == gt_img.size == edge_img.size:
                rgb_list.append(rgb_path)
                thermal_list.append(thermal_path)
                gt_list.append(gt_path)
                edge_list.append(edge_path)
        self.rgb_images = rgb_list
        self.thermal_images = thermal_list
        self.gts = gt_list
        self.edges = edge_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def Threshold_process(self, a):
        one = torch.ones_like(a)
        return torch.where(a > 0, one, a)

    def __len__(self):
        return self.size


# ==============================
# ==============================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader_Multi(rgb_root, thermal_root, gt_root, edge_root,
                     batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    dataset = PolypDataset_Multi(rgb_root, thermal_root, gt_root, edge_root, trainsize)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )
    return data_loader


# ==============================
# ==============================
class test_dataset_Multi:
    def __init__(self, rgb_root, thermal_root, gt_root, testsize):
        self.testsize = testsize
        self.rgb_images = sorted([rgb_root + f for f in os.listdir(rgb_root)
                                  if f.endswith('.jpg') or f.endswith('.png')])
        self.thermal_images = sorted([thermal_root + f for f in os.listdir(thermal_root)
                                      if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root)
                        #    if f.endswith('.tif') or f.endswith('.png')])
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.rgb_images)
        self.index = 0

    def load_data(self):
        rgb_img = self.rgb_loader(self.rgb_images[self.index])
        thermal_img = self.rgb_loader(self.thermal_images[self.index])
        rgb_img = self.transform(rgb_img).unsqueeze(0)
        thermal_img = self.transform(thermal_img).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.rgb_images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index = (self.index + 1) % self.size
        return rgb_img, thermal_img, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


if __name__ == '__main__':
    rgb_root = '/path/to/rgb/images/'
    thermal_root = '/path/to/thermal/images/'
    gt_root = '/path/to/gt/'
    edge_root = '/path/to/edge/'
    batchsize = 4
    trainsize = 512
    loader = get_loader_Multi(rgb_root, thermal_root, gt_root, edge_root,
                              batchsize, trainsize, num_workers=8)
    for i, (rgb, thermal, gt, edge) in enumerate(loader, start=1):
        print("Batch", i)
        print("RGB shape:", rgb.shape)
        print("Thermal shape:", thermal.shape)
        print("GT shape:", gt.shape)
        print("Edge shape:", edge.shape)
        break
