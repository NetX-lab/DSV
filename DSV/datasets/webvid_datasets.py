import dataclasses
# WebVid validation split
import os
import random
import sys
import time
from collections import deque
from typing import Optional

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import webdataset as wds
from DSV.datasets import video_transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm
from video2dataset.dataloader import get_video_dataset
from video2dataset.dataloader.transform import VideoResizer
from video2dataset.dataloader.video_decode import VideoDecorder
from webdataset import WebLoader



class WebVid(torch.utils.data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all, self.video_frame_all = self.load_video_frames(self.data_path)
        self.video_num = len(self.data_all)
        self.video_frame_num = len(self.video_frame_all)

        # sky video frames
        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num

        # video_transforms.ToTensorVideo(), # TCHW
        #     video_transforms.RandomHorizontalFlipVideo(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

        self.image_size = configs.image_size

        if self.image_size == 256:
            self.image_tranform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                    ),
                ]
            )
        else:
            self.image_tranform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                    ),
                ]
            )

    def __getitem__(self, index):
        video_index = index % self.video_num
        vframes = self.data_all[video_index]
        total_frames = len(vframes)

        # Sampling video frames

        select_video_frames = []

        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)

        frame_indice = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.target_video_len, dtype=int
        )
        # print(frame_indice)

        select_video_frames = vframes[
            frame_indice[0] : frame_indice[-1] + 1 : self.frame_interval
        ]

        video_frames = []
        for path in select_video_frames:
            image = Image.open(path).convert("RGB")
            video_frame = torch.as_tensor(
                np.array(image, dtype=np.uint8, copy=True)
            ).unsqueeze(0)
            video_frames.append(video_frame)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)

        # get video frames
        images = []
        for i in range(self.use_image_num):
            while True:
                try:
                    video_frame_path = self.video_frame_all[index + i]
                    image_path = os.path.join(self.data_path, video_frame_path)
                    image = Image.open(image_path).convert("RGB")
                    image = self.image_tranform(image).unsqueeze(0)

                    # image = torch.as_tensor(np.array(image, dtype=np.uint8, copy=True)).unsqueeze(0)
                    images.append(image)
                    break
                except Exception as e:
                    index = random.randint(0, self.video_frame_num - self.use_image_num)
                    print(
                        f"Fail to load image: {image_path}, try to load another image"
                    )

        images = torch.cat(images, dim=0)
        # images = self.transform(images)
        assert len(images) == self.use_image_num

        # print(f"video_clip.shape: {video_clip.shape}, images.shape: {images.shape}")

        video_cat = torch.cat([video_clip, images], dim=0)

        # print(f"video clip shape: {video_clip.shape}, images shape: {images.shape}, video_cat shape: {video_cat.shape}")

        return {"video": video_cat, "video_name": 1}

    def __len__(self):
        return self.video_frame_num

    def load_video_frames(self, dataroot):
        data_all = []
        frames_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                frames = sorted(
                    meta[2], key=lambda item: int(item.split(".")[0].split("_")[-1])
                )
            except:
                print(meta[0], meta[2])
            frames = [
                os.path.join(root, item) for item in frames if is_image_file(item)
            ]
            # if len(frames) > max(0, self.sequence_length * self.sample_every_n_frames):
            if len(frames) != 0:
                data_all.append(frames)
                for frame in frames:
                    frames_all.append(frame)
        # self.video_num = len(data_all)
        return data_all, frames_all


def enumerate_report(seq, delta, growth=1.0):
    last = 0
    count = 0
    for count, item in enumerate(seq):
        now = time.time()
        if now - last > delta:
            last = now
            yield count, item, True
        else:
            yield count, item, False
        delta *= growth


def make_dataloader_train():
    """Create a DataLoader for training on the ImageNet dataset using WebDataset."""

    SHARDS = "/mnt/xtan-jfs/webvid-10M/dataset/{00000..00002}.tar"
    decoder_kwargs = {
        "n_frames": 8,  # get 8 frames from each video
        "fps": 10,  # downsample to 10 FPS
        "num_threads": 12,  # use 12 threads to decode the video
    }
    resize_size = crop_size = 256
    batch_size = 4

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    def make_sample(sample):
        return transform(sample["mp4"]), sample["txt"]

    # This is the basic WebDataset definition: it starts with a URL and add shuffling,
    # decoding, and augmentation. Note `resampled=True`; this is essential for
    # distributed training to work correctly.
    # trainset = wds.WebDataset(trainset_url, resampled=True, cache_dir=cache_dir, nodesplitter=wds.split_by_node)

    trainset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
    )

    # trainset = trainset.shuffle(1000).decode("pil").map(make_sample)

    # For IterableDataset objects, the batching needs to happen in the dataset.
    trainset = trainset.batched(64)
    trainloader = wds.WebLoader(trainset, batch_size=None, num_workers=4)

    # We unbatch, shuffle, and rebatch to mix samples from different workers.
    trainloader = trainloader.unbatched().shuffle(1000).batched(batch_size)

    # A resampled dataset is infinite size, but we can recreate a fixed epoch length.
    trainloader = trainloader.with_epoch(1282 * 100 // 64)

    print("trainloader", trainloader)

    return trainloader


def make_dataloader(split="train"):
    """Make a dataloader for training or validation."""
    if split == "train":
        return make_dataloader_train()
    elif split == "val":
        return make_dataloader_val()  # not implemented for this notebook
    else:
        raise ValueError(f"unknown split {split}")


def reassemble(x):
    """
    Process a dictionary by updating its values based on certain conditions.

    :param dict x: The input dictionary to process.
    :return: The processed dictionary.
    :rtype: dict
    """

    new_dict = {}

    for key in x:
        if key not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            continue

        # this is updating the output of video decoders
        if isinstance(x[key], tuple) and len(x[key]) == 2:
            new_dict.update({f"{subk}": x[key][-1][subk] for subk in x[key][-1]})

        x[key] = x[key][0]
    x.update(new_dict)
    del new_dict
    return x


transform_webvid = transforms.Compose(
    [
        # transforms.ConvertImageDtype(torch.uint8),
        video_transforms.ToTensorVideo(),  # TCHW
        # video_transforms.RandomHorizontalFlipVideo(1),
        # video_transforms.UCFCenterCropVideo(256),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]
)


def video_transform(x):
    for key in x:
        if key in ["mp4", "ogv", "mjpeg", "avi", "mov", "h264", "mpg", "webm", "wmv"]:
            x[key] = x[key].to(torch.uint8).float() / 255.0
            # T,H,W,C
            x[key] = x[key].flip(-2) if random.random() < 0.5 else x[key]
            # T,C,H,W
            x[key] = x[key].permute(0, 3, 1, 2)
            x[key] = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            )(x[key])
            x[key] = x[key].permute(0, 2, 3, 1)  # T,H,W,C
    return x


if __name__ == "__main__":
    # dist.init_process_group(backend="nccl")
    # os.environ["GOPEN_VERBOSE"] = "1"
    # sample = next(iter(make_dataloader()))
    # print(sample[0].shape, sample[1].shape)
    # os.environ["GOPEN_VERBOSE"] = "0"

    import cv2

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()

    url = "xxx/webvid-10M/dataset/{00012..00018}.tar"

    pil_dataset = wds.WebDataset(url, nodesplitter=wds.split_by_node).shuffle(1000)

    decoder_kwargs = {
        "n_frames": 40,  # get 8 frames from each video
        "fps": 10,  # downsample to 10 FPS
        "num_threads": 12,  # use 12 threads to decode the video
    }

    handler = wds.warn_and_continue  # wds.reraise_exception
    transform_handler = wds.reraise_exception

    resize_size = crop_size = 256

    batch_size = 4

    pil_dataset = (
        pil_dataset.decode(
            VideoDecorder(**decoder_kwargs),
            handler=handler,
        )
        .map(reassemble, handler=handler)
        .map(
            VideoResizer(
                size=resize_size,
                crop_size=crop_size,
                random_crop=False,
                key="mp4",
                width_key="original_width",
                height_key="original_height",
            ),
            handler=handler,
        )
        .map(video_transform, handler=transform_handler)
        .to_tuple("mp4", "txt")
    )

    # dataset=torch.utils.data.DataLoader((pil_dataset), batch_size=batch_size)

    dataloader = WebLoader(
        pil_dataset, batch_size=batch_size, num_workers=4
    ).with_length(100000)

    # sampler=DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

    # dataloader=DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # dataloader=WebLoader(pil_dataset, batch_size=batch_size, num_workers=4)

    for sample in dataloader:
        video, caption = sample
        print(f"rank: {rank}, video shape: {video.shape}, caption: {caption}")
        break

    # video is a list of frame tensors, each tensor is a frame of shape [H,W,3]. Save the video tensor list into a video file
    import imageio

    # Assuming video is a list of frame tensors, each tensor is a frame of shape [H,W,3]
    # Save the video tensor list into a video file
    def save_video(video, output_path):
        """
        Save a list of frame tensors into a video file.

        :param video: List of frame tensors, each tensor is a frame of shape [H,W,3]
        :param output_path: Path to save the output video file
        """
        with imageio.get_writer(output_path, fps=10) as writer:
            for frame in video:
                writer.append_data(frame)

    # Example usage
    output_video_path = "output_video.mp4"

    save_video((video[1].numpy() * 255).astype(np.uint8), output_video_path)
