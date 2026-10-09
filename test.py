import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lib.Network_Multimodal_LogitsGatedFusion import Multi_Network


class TestDataset(Dataset):
    def __init__(self, rgb_folder, thermal_folder, gt_folder, testsize):
        extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        gt_extensions = extensions + ('.tif',)

        self.rgb_images = sorted(
            os.path.join(rgb_folder, name)
            for name in os.listdir(rgb_folder)
            if name.lower().endswith(extensions)
        )
        self.thermal_images = sorted(
            os.path.join(thermal_folder, name)
            for name in os.listdir(thermal_folder)
            if name.lower().endswith(extensions)
        )
        self.ground_truths = sorted(
            os.path.join(gt_folder, name)
            for name in os.listdir(gt_folder)
            if name.lower().endswith(gt_extensions)
        )

        if not (
            len(self.rgb_images)
            == len(self.thermal_images)
            == len(self.ground_truths)
        ):
            raise ValueError(
                'RGB, thermal, and ground-truth folders must contain the same '
                f'number of images; got {len(self.rgb_images)}, '
                f'{len(self.thermal_images)}, and {len(self.ground_truths)}.'
            )

        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        with Image.open(self.rgb_images[index]) as image:
            rgb = self.transform(image.convert('RGB'))
        with Image.open(self.thermal_images[index]) as image:
            thermal = self.transform(image.convert('RGB'))
        with Image.open(self.ground_truths[index]) as image:
            gt = np.asarray(image.convert('L'), dtype=np.float32)

        name = os.path.basename(self.rgb_images[index])
        if name.lower().endswith(('.jpg', '.jpeg')):
            name = os.path.splitext(name)[0] + '.png'
        return rgb, thermal, gt, name


def collate_batch(batch):
    rgb, thermal, gt, names = zip(*batch)
    return torch.stack(rgb), torch.stack(thermal), list(gt), list(names)


def compute_mae(prediction, ground_truth):
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
    ground_truth = np.clip(np.asarray(ground_truth, dtype=np.float32), 0.0, 1.0)
    return float(np.mean(np.abs(prediction - ground_truth)))


def evaluate(model, data_loader, device, results_dir):
    model.eval()
    total_mae = 0.0
    image_count = 0
    os.makedirs(results_dir, exist_ok=True)

    with torch.no_grad():
        for rgb, thermal, gt_batch, names in data_loader:
            rgb = rgb.to(device, non_blocking=True)
            thermal = thermal.to(device, non_blocking=True)
            outputs = model(rgb, thermal)
            fused_logits = outputs[18]

            for index, (gt, name) in enumerate(zip(gt_batch, names)):
                gt = np.asarray(gt, dtype=np.float32)
                gt = gt / (gt.max() + 1e-8)

                prediction = torch.sigmoid(
                    F.interpolate(
                        fused_logits[index:index + 1],
                        size=gt.shape,
                        mode='bilinear',
                        align_corners=False,
                    )
                )
                prediction = prediction.cpu().numpy().squeeze().astype(np.float32)
                prediction = (prediction - prediction.min()) / (
                    prediction.max() - prediction.min() + 1e-8
                )
                prediction_image = Image.fromarray(
                    (prediction * 255).astype(np.uint8), mode='L'
                )
                prediction_image.save(os.path.join(results_dir, name))

                mae = compute_mae(prediction, gt)
                print(f'{name}: MAE={mae:.6f}')
                total_mae += mae
                image_count += 1

    if image_count == 0:
        raise ValueError('The test dataset is empty.')

    average_mae = total_mae / image_count
    print(f'Average MAE={average_mae:.6f}')
    return average_mae


def main():
    parser = argparse.ArgumentParser(description='Evaluate RGB-T COD using MAE.')
    parser.add_argument('--testsize', type=int, default=384)
    parser.add_argument('--pth_path', type=str, required=True)
    parser.add_argument('--rgb_folder', type=str, required=True)
    parser.add_argument('--thermal_folder', type=str, required=True)
    parser.add_argument('--gt_folder', type=str, required=True)
    parser.add_argument(
        '--results_dir',
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
    )
    parser.add_argument('--batch_size', type=int, default=4)
    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Multi_Network(channels=96)
    state_dict = torch.load(opt.pth_path, map_location='cpu')
    state_dict = {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model = model.to(device)

    dataset = TestDataset(
        opt.rgb_folder,
        opt.thermal_folder,
        opt.gt_folder,
        opt.testsize,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        pin_memory=device.type == 'cuda',
        collate_fn=collate_batch,
    )

    evaluate(model, data_loader, device, opt.results_dir)


if __name__ == '__main__':
    main()
