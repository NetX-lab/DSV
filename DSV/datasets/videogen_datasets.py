import io
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import DSV.datasets.video_transforms as video_transforms

# we have a json file, which contains the video id and the viedo text; And another folder contains all video mp4 files, the name of the video is the video id.
# we want to load the video and the text, and return the video and the text. Please implement the video dataset class.


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
            size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        """
        Args:
                total_frames (int): Total number of frames in the video
        Returns:
                tuple: (begin_index, end_index) The start and end frame indices of the cropped video
        """
        # Ensure total frames is greater than or equal to required frames
        if total_frames < self.size:
            begin_index = 0
            end_index = total_frames
        else:
            # Ensure end_index - begin_index == self.size
            rand_end = total_frames - self.size
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + self.size

        return begin_index, end_index


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


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class VideoGenDataset(data.Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.json_file = configs.json_file
        self.video_folder = configs.data_path
        self.transform = get_transforms_video(configs.image_size)
        self.temporal_sample = TemporalRandomCrop(
            size=self.configs.num_frames * self.configs.frame_interval
        )
        self.target_video_len = self.configs.num_frames
        self.data = self.load_data()

    def load_data(self):
        with open(self.json_file, "r") as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def load_video(self, video_path):
        # Use torchvision's read_video function to read video

        vframes, aframes, info = torchvision.io.read_video(
            filename=video_path, pts_unit="sec", output_format="TCHW"
        )

        return vframes

    def __getitem__(self, index):
        video_id, text = self.data[index]["vid"], self.data[index]["caption"]
        video_path = os.path.join(self.video_folder, f"{video_id}.mp4")

        try:
            video = self.load_video(video_path)

            if video.shape[1] != 3 or video.shape[2] <= 0 or video.shape[3] <= 0:
                raise Exception(f"The video shape is not correct: {video.shape}")

        except Exception as e:
            print(f"Error loading video {video_path}: {e}; Workaround with index+1")
            return self.__getitem__(index + 1)

        total_frames = len(video)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)

        if (end_frame_ind - start_frame_ind) < self.target_video_len:
            print(
                f"Error!!!! rank:{dist.get_rank()}; index:{index}; total frames: {total_frames}, start_frame_ind: {start_frame_ind}, end_frame_ind: {end_frame_ind}, file path: {video_path}; workaround with index+1"
            )
            index += 1
            return self.__getitem__(index)
        else:
            pass

        frame_indice = np.linspace(
            start_frame_ind,
            end_frame_ind - 1,
            self.target_video_len,
            dtype=int,
            endpoint=True,
        )
        video = video[frame_indice]

        try:
            video = self.transform(video)
        except Exception as e:
            print(f"Error transforming video: {e}; workaround with index+1")
            return self.__getitem__(index + 1)

        return {"video": video, "video_text_prompt": text}


class GroupedDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, world_size=8, ranks_per_group=2, shuffle=True, seed=0):
        self.dataset = dataset
        self.world_size = world_size
        self.ranks_per_group = ranks_per_group
        self.num_groups = world_size // ranks_per_group
        self.rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        )
        self.group_id = self.rank // ranks_per_group
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Ensure the dataset length can be divided by the number of groups
        self.num_samples = len(dataset) // self.num_groups
        if len(dataset) % self.num_groups != 0:
            self.num_samples += 1
        self.total_size = self.num_samples * self.num_groups

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Make up data
        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]

        # Important modification: Rearrange indices by group, but keep data within groups consistent
        indices = [
            indices[i : i + self.num_groups]
            for i in range(0, len(indices), self.num_groups)
        ]

        # Calculate rank within group
        rank_in_group = self.rank % self.ranks_per_group
        # Get all data for current group
        indices = [batch[self.group_id] for batch in indices]

        if rank_in_group == 0:
            return iter(indices)
        else:
            return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    config_dict = {
        "json_file": "xx/VidGen_1M_video_caption.json",
        "video_folder": "xx/VIDGEN-1M",
        "data_path": "xx/VIDGEN-1M",
        "num_frames": 48,
        "frame_interval": 2,
        "image_size": 512,
    }

    configs = Config(**config_dict)

    group_size = dist.get_world_size() // 2

    dataset = VideoGenDataset(configs)
    sampler = GroupedDistributedSampler(
        dataset,
        world_size=dist.get_world_size(),
        ranks_per_group=group_size,
        shuffle=True,
        seed=0,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler, num_workers=6
    )

    # make a shared dict where each process can write to it

    this_rank_dict = {}

    for i, data in enumerate(dataloader):
        if i == 200:
            break
        video = data["video"]
        text = data["video_text_prompt"]
        print(f"rank {dist.get_rank()}: batch {i} video shape {video.shape}")
        print(f"rank {dist.get_rank()}: batch {i} text {text}")
        this_rank_dict[i] = text

    # share the dict to all processes
    output_dict = {}
    dist.all_gather_object(output_dict, this_rank_dict)

    if dist.get_rank() == 0:
        output_0 = output_dict[0]
        output_1 = output_dict[1]
        for (k, v), (k1, v1) in zip(output_0.items(), output_1.items()):
            assert v == v1

        output_2 = output_dict[2]
        output_3 = output_dict[3]
        for (k, v), (k1, v1) in zip(output_2.items(), output_3.items()):
            assert v == v1

        print("test passed!!!")

    dist.destroy_process_group()
