from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DATASETS = {
    'office-31': (('amazon', 'dslr', 'webcam'), 31),
    'office-home': (('Art', 'Clipart', 'Product', 'Real_World'), 65),
}


class OfficeDataset(Dataset):
    def __init__(self, root, task, split, dataset='office-31'):
        self.root = Path(root)
        manifest = Path(__file__).parent / 'data_txt' / dataset / f'{task}_{split}.txt'
        self.records = [line.rsplit(' ', 1) for line in manifest.read_text().splitlines() if line.strip()]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        relative, label = self.records[index]
        with Image.open(self.root / relative) as image:
            return self.transform(image.convert('RGB')), int(label)


def make_loaders(root, dataset, batch_size, workers):
    tasks, _ = DATASETS[dataset]
    loaders = {}
    for split in ('train', 'val', 'test'):
        loaders[split] = {}
        for task in tasks:
            loader = DataLoader(OfficeDataset(root, task, split, dataset),
                                batch_size=batch_size, shuffle=split == 'train',
                                drop_last=split == 'train', num_workers=workers,
                                pin_memory=True)
            if len(loader) == 0:
                raise ValueError(f'{task}/{split} has no batches; reduce batch size')
            loaders[split][task] = loader
    return loaders


def next_batch(loaders, iterators, task):
    try:
        return next(iterators[task])
    except StopIteration:
        iterators[task] = iter(loaders[task])
        return next(iterators[task])
