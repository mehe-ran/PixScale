import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class HighResDataset(Dataset):
    def __init__(self, image_dir, crop_size=96, scale_factor=4):
        self.image_dir = image_dir

        # recursive scan to find images buried in subfolders
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_files.append(os.path.join(root, f))

        # high res target (96x96)
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
        ])

        # low res input (24x24)
        self.lr_transform = transforms.Compose([
            transforms.Resize(crop_size // scale_factor, interpolation=transforms.InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # skip corrupted files safely
            return self.__getitem__((idx + 1) % len(self.image_files))

        hr_image = self.hr_transform(img)
        lr_image = self.lr_transform(hr_image)

        # Standard [0, 1] pixel values. No negative math.
        return lr_image, hr_image