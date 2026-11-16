from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms


def image_filenames(root):
    """List only supported files directly inside a flat image directory."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Image folder does not exist: {root}')
    return sorted(
        (p.name for p in root.iterdir()
         if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}),
        key=lambda name: (name.casefold(), name),
    )


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None, files=None):
        super(ImageFolderDataset, self).__init__()
        self.root = Path(root)
        self.transform = transform
        available = image_filenames(self.root)
        self.files = available if files is None else list(files)
        if len(self.files) != len(set(self.files)):
            raise ValueError(f'Duplicate image filenames supplied for {root}')
        missing = set(self.files) - set(available)
        if missing:
            raise ValueError(f'Invalid/missing image filenames in {root}: {sorted(missing)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.root / self.files[idx]
        try:
            with Image.open(image_path) as source:
                image = source.convert('RGB')
        except (OSError, ValueError) as exc:
            raise RuntimeError(f'Cannot read image {image_path}: {exc}') from exc

        if self.transform:
            image = self.transform(image)

        return image


def get_transform(size, crop, final_size, deterministic=False):
    if size <= 0:
        raise ValueError('size must be positive')
    if final_size <= 0:
        raise ValueError('final_size must be positive')
    transform_list = []
    if crop:
        # Integer Resize preserves aspect ratio and resizes the shorter edge.
        transform_list.append(transforms.Resize(max(size, final_size), antialias=True))
        crop_type = transforms.CenterCrop if deterministic else transforms.RandomCrop
        transform_list.append(crop_type(final_size))
    else:
        transform_list.append(transforms.Resize((final_size, final_size), antialias=True))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        

def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std
