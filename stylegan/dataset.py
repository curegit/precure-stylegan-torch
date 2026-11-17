from os.path import lexists, isdir
from bisect import bisect_right
from itertools import accumulate
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from utilities.stdio import eprint
from utilities.image import load_image, load_image_uint8, uint8_to_float
from utilities.filesys import relaxed_glob_recursively


class ImageDataset(Dataset):

    extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"]

    def __init__(self, directory, resolution):
        super().__init__()
        self.resolution = resolution
        self.preloaded = False
        if not lexists(directory):
            eprint(f"Invalid dataset: {directory}")
            eprint("No such directory!")
            raise RuntimeError("Input error")
        if not isdir(directory):
            eprint(f"Invalid dataset: {directory}")
            eprint("Specified path is not a correct directory!")
            raise RuntimeError("Input error")
        self.image_files = sorted(sum([relaxed_glob_recursively(directory, e) for e in ImageDataset.extensions], []))
        if not self.image_files:
            eprint(f"Invalid dataset: {directory}")
            eprint("No images found in the directory!")
            raise RuntimeError("Input error")

    def __len__(self):
        return len(self.image_files)

    def preload(self, callback=None):
        images = []
        for i in self.image_files:
            images.append(load_image_uint8(i, self.resolution))
            if callback:
                callback()
        self.loaded_images = images
        self.preloaded = True

    def __getitem__(self, index):
        image = uint8_to_float(self.loaded_images[index]) if self.preloaded else load_image(self.image_files[index], self.resolution)
        return torch.from_numpy(image).contiguous()

    def get_example(self, index):
        return self[index]


class MulticategoryImageDataset(Dataset):

    def __init__(self, directories, resolution):
        super().__init__()
        self.datasets = [ImageDataset(d, resolution) for d in directories]
        if not self.datasets:
            raise ValueError("At least one dataset directory is required")
        self.ends = list(accumulate(len(dataset) for dataset in self.datasets))
        self.conditions = torch.eye(len(self.datasets))

    def __len__(self):
        return self.ends[-1]

    def preload(self, callback=None):
        for dataset in self.datasets:
            dataset.preload(callback)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        category = bisect_right(self.ends, index)
        offset = self.ends[category - 1] if category else 0
        return self.datasets[category][index - offset], self.conditions[category].clone()

    def get_example(self, index):
        return self[index]


class TrainingIterator(Sampler):
    """Shuffled, full batches across epoch boundaries, with resumable ordering.

    Like the former SerialIterator, the last batch wraps into the next epoch;
    no images are dropped and minibatch/accumulation group sizes stay constant.
    Loading runs in the main process so the saved cursor has no prefetched data.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        if not len(dataset) or batch_size < 1:
            raise ValueError("A nonempty dataset and positive batch size are required")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = torch.Generator().manual_seed(torch.randint(2**63 - 1, ()).item())
        self.position = 0
        self.images_seen = 0
        self.order = self.new_order()
        loader = DataLoader(dataset, batch_sampler=self, num_workers=0, generator=torch.Generator().manual_seed(0))
        self.loader_iterator = iter(loader)

    def new_order(self):
        return torch.randperm(len(self.dataset), generator=self.rng) if self.shuffle else torch.arange(len(self.dataset))

    def __iter__(self):
        while True:
            indices = []
            while len(indices) < self.batch_size:
                if self.position == len(self.dataset):
                    self.order = self.new_order()
                    self.position = 0
                count = min(self.batch_size - len(indices), len(self.dataset) - self.position)
                indices.extend(self.order[self.position : self.position + count].tolist())
                self.position += count
            self.images_seen += len(indices)
            yield indices

    def next(self):
        return next(self.loader_iterator)

    @property
    def epoch_detail(self):
        return self.images_seen / len(self.dataset)

    def state_dict(self):
        return {"length": len(self.dataset), "batch_size": self.batch_size, "shuffle": self.shuffle, "position": self.position, "images_seen": self.images_seen, "order": self.order.clone(), "rng": self.rng.get_state()}

    def load_state_dict(self, state):
        if (state["length"], state["batch_size"], state["shuffle"]) != (len(self.dataset), self.batch_size, self.shuffle):
            raise ValueError("Snapshot dataset length, batch size or shuffle setting does not match")
        self.position = state["position"]
        self.images_seen = state["images_seen"]
        self.order = state["order"].clone()
        self.rng.set_state(state["rng"])
