import torch
import os

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import logging
import webdataset as wds
from torch.utils.data.dataset import IterableDataset, ChainDataset
from torch.utils.data import Dataset, ConcatDataset
from typing import Iterable

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)


def prepare_sample(samples, cuda_enabled=True):
    if cuda_enabled:
        samples = move_to_cuda(samples)

    # TODO fp16 support

    return samples

def reorg_datasets_by_split(datasets):
    """
    Organizes datasets by split.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by name.

    Returns:
        Dict of datasets by split {split_name: List[Datasets]}.
    """
    # if len(datasets) == 1:
    #     return datasets[list(datasets.keys())[0]]
    # else:
    reorg_datasets = dict()

    # reorganize by split
    for _, dataset in datasets.items():
        for split_name, dataset_split in dataset.items():
            if split_name not in reorg_datasets:
                reorg_datasets[split_name] = [dataset_split]
            else:
                reorg_datasets[split_name].append(dataset_split)

    return reorg_datasets


def concat_datasets(datasets):
    """
    Concatenates multiple datasets into a single dataset.

    It supports may-style datasets and DataPipeline from WebDataset. Currently, does not support
    generic IterableDataset because it requires creating separate samplers.

    Now only supports conctenating training datasets and assuming validation and testing
    have only a single dataset. This is because metrics should not be computed on the concatenated
    datasets.

    Args:
        datasets: dict of torch.utils.data.Dataset objects by split.

    Returns:
        Dict of concatenated datasets by split, "train" is the concatenation of multiple datasets,
        "val" and "test" remain the same.

        If the input training datasets contain both map-style and DataPipeline datasets, returns
        a tuple, where the first element is a concatenated map-style dataset and the second
        element is a chained DataPipeline dataset.

    """
    # concatenate datasets in the same split
    for split_name in datasets:
        if split_name != "train":
            assert (
                len(datasets[split_name]) == 1
            ), "Do not support multiple {} datasets.".format(split_name)
            datasets[split_name] = datasets[split_name][0]
        else:
            iterable_datasets, map_datasets = [], []
            for dataset in datasets[split_name]:
                if isinstance(dataset, wds.DataPipeline):
                    logging.info(
                        "Dataset {} is IterableDataset, can't be concatenated.".format(
                            dataset
                        )
                    )
                    iterable_datasets.append(dataset)
                elif isinstance(dataset, IterableDataset):
                    raise NotImplementedError(
                        "Do not support concatenation of generic IterableDataset."
                    )
                else:
                    map_datasets.append(dataset)

            # if len(iterable_datasets) > 0:
            # concatenate map-style datasets and iterable-style datasets separately
            chained_datasets = (
                ChainDataset(iterable_datasets) if len(iterable_datasets) > 0 else None
            )
            concat_datasets = (
                ConcatDataset(map_datasets) if len(map_datasets) > 0 else None
            )

            train_datasets = concat_datasets, chained_datasets
            train_datasets = tuple([x for x in train_datasets if x is not None])
            train_datasets = (
                train_datasets[0] if len(train_datasets) == 1 else train_datasets
            )

            datasets[split_name] = train_datasets

    return datasets


class SubjectDrivenTextToImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        subject_text,
        inp_image_processor,
        tgt_image_processor,
        txt_processor,
        repetition=100000,
    ):
        self.subject = txt_processor(subject_text.lower())
        self.image_dir = image_dir

        self.inp_image_transform = inp_image_processor
        self.tgt_image_transform = tgt_image_processor

        self.text_processor = txt_processor

        image_paths = os.listdir(image_dir)
        # image paths are jpg png webp
        image_paths = [
            os.path.join(image_dir, imp)
            for imp in image_paths
            if os.path.splitext(imp)[1][1:]
            in ["jpg", "png", "webp", "jpeg", "JPG", "PNG", "WEBP", "JPEG"]
        ]
        # make absolute path
        self.image_paths = [os.path.abspath(imp) for imp in image_paths]
        self.repetition = repetition

    def __len__(self):
        return len(self.image_paths) * self.repetition
    
    @property
    def len_without_repeat(self):
        return len(self.image_paths)

    def collater(self, samples):
        return default_collate(samples)

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert("RGB")

        # For fine-tuning, we use the same caption for all images
        # maybe worth trying different captions for different images
        caption = f"a {self.subject}"
        caption = self.text_processor(caption)

        inp_image = self.inp_image_transform(image)
        tgt_image = self.tgt_image_transform(image)

        return {
            "inp_image": inp_image,
            "tgt_image": tgt_image,
            "caption": caption,
            "subject_text": self.subject,
        }
    

class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
