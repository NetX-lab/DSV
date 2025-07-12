import csv
import json
import os

import ipdb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from . import video_transforms


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    return transform


class OpenVidDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        config,
    ):
        video_samples = []

        self.video_folder = config.data_path

        # with open(config.csv_path, "r") as f:
        #     for i,row in enumerate(csv.reader(f)):
        #         if i==0:
        #             continue

        #         vid_path = os.path.join(self.video_folder, row[0])
        #         if os.path.exists(vid_path):
        #             video_samples.append([vid_path, row[1]])

        with open(config.json_path, "r") as f:
            video_samples = json.load(f)

        print(f"init the dataset with {len(video_samples)} samples")

        self.samples = video_samples

        self.is_video = True
        self.transform = get_transforms_video()
        self.num_frames = config.num_frames
        self.frame_interval = config.frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(
            self.num_frames * self.frame_interval
        )

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        text = sample[1]

        path = os.path.join(self.video_folder, path)

        if self.is_video:
            is_exit = os.path.exists(path)
            if is_exit:
                vframes, aframes, info = torchvision.io.read_video(
                    filename=path, pts_unit="sec", output_format="TCHW"
                )
                total_frames = len(vframes)
            else:
                total_frames = 0

            #  video exits and total_frames >= self.num_frames

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(
                start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int
            )

            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        # video = video.permute(1, 0, 2, 3)

        return {"video": video, "video_text_prompt": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    data_path = ""
    root = ""
    dataset = OpenVidDataset(config)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=1)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    for video_data in loader:
        print(video_data)
