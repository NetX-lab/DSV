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


class DummyDataset(data.Dataset):
    """
    Dummy Video Generation Dataset that generates fake video data instead of loading real videos.
    Maintains the same interface as VideoGenDataset but generates synthetic data.
    """
    
    def __init__(self, configs):
        self.configs = configs
        self.image_size = getattr(configs, 'image_size', 256)
        self.num_frames = getattr(configs, 'num_frames', 16)
        self.frame_interval = getattr(configs, 'frame_interval', 1)
        self.dataset_size = getattr(configs, 'dataset_size', 10000)  # Number of fake samples
        
        # Generate fake text prompts
        self.text_prompts = self._generate_text_prompts()
        
        # Set random seed for reproducibility
        self.seed = getattr(configs, 'seed', 42)
        
    def _generate_text_prompts(self):
        """Generate a pool of fake text prompts for videos"""
        base_prompts = [
            "A cat playing in the garden",
            "Beautiful sunset over the ocean",
            "People walking in a busy city street",
            "A dog running in the park",
            "Clouds moving across the sky",
            "A child riding a bicycle",
            "Birds flying over a lake",
            "Traffic flowing on a highway",
            "Flowers blooming in spring",
            "A train passing through countryside",
            "Rain falling on the window",
            "A dancer performing on stage",
            "Waves crashing on the beach",
            "A chef cooking in the kitchen",
            "Snow falling in winter forest",
            "A musician playing guitar",
            "Fireflies glowing at night",
            "A boat sailing on calm water",
            "Children playing in playground",
            "Lightning striking in storm"
        ]
        
        # Extend the prompts list to cover the dataset size
        extended_prompts = []
        for i in range(self.dataset_size):
            base_prompt = base_prompts[i % len(base_prompts)]
            # Add some variation
            variations = [
                f"{base_prompt}",
                f"{base_prompt} in slow motion",
                f"{base_prompt} at sunset",
                f"{base_prompt} in high definition",
                f"Beautiful {base_prompt.lower()}",
            ]
            extended_prompts.append(variations[i % len(variations)])
            
        return extended_prompts

    def __len__(self):
        return self.dataset_size

    def _generate_fake_video(self, index):
        """Generate a fake video tensor with the specified dimensions"""
        # Set seed based on index for reproducible generation
        torch.manual_seed(self.seed + index)
        np.random.seed(self.seed + index)
        
        # Generate fake video: [T, C, H, W]
        # T = num_frames, C = 3 (RGB), H = W = image_size
        video = torch.randn(
            self.num_frames, 
            3, 
            self.image_size, 
            self.image_size,
            dtype=torch.float32
        )
        
        # Apply some patterns to make it look more video-like
        # Add temporal consistency - each frame is slightly similar to the previous one
        for t in range(1, self.num_frames):
            video[t] = 0.7 * video[t] + 0.3 * video[t-1]
        
        # Add some spatial patterns
        h, w = self.image_size, self.image_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        # Create different patterns based on index
        pattern_type = index % 4
        if pattern_type == 0:
            # Circular pattern
            center_y, center_x = h // 2, w // 2
            pattern = torch.exp(-((y - center_y)**2 + (x - center_x)**2) / (h * w * 0.1))
        elif pattern_type == 1:
            # Wave pattern
            pattern = torch.sin(2 * np.pi * x / w * 3) * torch.cos(2 * np.pi * y / h * 3)
        elif pattern_type == 2:
            # Gradient pattern
            pattern = (x + y) / (h + w)
        else:
            # Checkerboard pattern
            pattern = ((x // 16) + (y // 16)) % 2
        
        # Apply pattern to all frames and channels
        for t in range(self.num_frames):
            for c in range(3):
                video[t, c] += 0.3 * pattern
        
        # Normalize to [-1, 1] range (matching the real dataset normalization)
        video = torch.clamp(video, -1, 1)
        
        return video

    def __getitem__(self, index):
        # Handle index overflow
        index = index % self.dataset_size
        
        # Generate fake video data
        video = self._generate_fake_video(index)
        
        # Get corresponding text prompt
        text = self.text_prompts[index]
        
        return {"video": video, "video_text_prompt": text}


class GroupedDistributedSampler(torch.utils.data.Sampler):
    """Reuse the same GroupedDistributedSampler from the original implementation"""
    
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


# Example usage and testing function
def test_dummy_dataset():
    """Test function to verify the dummy dataset works correctly"""
    
    # Create a config object
    config = Config(
        image_size=256,
        num_frames=16,
        frame_interval=1,
        dataset_size=1000,
        seed=42
    )
    
    # Create dummy dataset
    dataset = DummyVideoGenDataset(config)
    
    # Test basic functionality
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    video = sample["video"]
    text = sample["video_text_prompt"]
    
    print(f"Video shape: {video.shape}")
    print(f"Text prompt: {text}")
    print(f"Video dtype: {video.dtype}")
    print(f"Video value range: [{video.min():.3f}, {video.max():.3f}]")
    
    # Test multiple samples
    for i in range(5):
        sample = dataset[i]
        print(f"Sample {i}: {sample['video_text_prompt']}")
    
    print("Dummy dataset test completed successfully!")


if __name__ == "__main__":
    test_dummy_dataset()