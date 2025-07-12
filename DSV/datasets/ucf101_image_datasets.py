import io
import json
import os
import re

import decord
import numpy as np
import torch
import torchvision
from wordsegment import load, segment

load()


import random
import traceback
from typing import Dict, List, Tuple

import torch.distributed as dist
from einops import rearrange
from PIL import Image
from torchvision import transforms

class_labels_map = None
cls_sample_cnt = None

class_labels_map = None
cls_sample_cnt = None


# def temporal_sampling(frames, start_idx, end_idx, num_samples):
#     """
#     Given the start and end frame index, sample num_samples frames between
#     the start and end with equal interval.
#     Args:
#         frames (tensor): a tensor of video frames, dimension is
#             `num video frames` x `channel` x `height` x `width`.
#         start_idx (int): the index of the start frame.
#         end_idx (int): the index of the end frame.
#         num_samples (int): number of frames to sample.
#     Returns:
#         frames (tersor): a tensor of temporal sampled video frames, dimension is
#             `num clip frames` x `channel` x `height` x `width`.
#     """
#     index = torch.linspace(start_idx, end_idx, num_samples)
#     index = torch.clamp(index, 0, frames.shape[0] - 1).long()
#     frames = torch.index_select(frames, 0, index)
#     return frames


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
                total_frames (int): 视频总帧数
        Returns:
                tuple: (begin_index, end_index) 裁剪的起始和结束帧索引
        """
        # 确保总帧数大于等于所需帧数
        if total_frames < self.size:
            begin_index = 0
            end_index = total_frames
        else:
            # 确保end_index - begin_index == self.size
            rand_end = total_frames - self.size
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + self.size

        return begin_index, end_index


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append( filename)
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth="./k400_classmap.json"):
    global class_labels_map, cls_sample_cnt

    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, "r") as fin:
        for line in fin:
            line_split = line.strip().split("\t")
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample["video"] = frame_dir
            idx += 1

            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f"missing label in line: {line}"
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])

            # choose a class subset of whole dataset
            if class_index < num_class:
                sample["label"] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name] += 1

    return dataset


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"
            f"num_threads={self.num_threads})"
        )
        return repr_str


class UCF101Images(torch.utils.data.Dataset):
    """Load the UCF101 video files

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self, configs, transform=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.video_lists = get_filelist(configs.data_path)
        self.transform = transform
        self.temporal_sample = TemporalRandomCrop(
            self.configs.num_frames * self.configs.frame_interval
        )
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        self.classes, self.class_to_idx = find_classes(self.data_path)
        self.video_num = len(self.video_lists)

        # ucf101 video frames
        self.frame_data_path = configs.frame_data_path  # important

        self.video_frame_txt = configs.frame_data_txt
        self.video_frame_files = [
            frame_file.strip() for frame_file in open(self.video_frame_txt)
        ]
        random.shuffle(self.video_frame_files)
        self.use_image_num = configs.use_image_num
        self.image_tranform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        self.video_frame_num = len(self.video_frame_files)

        print(f"class_to_idx: {self.class_to_idx}")
        print(f"video clips num: {self.video_num}")

        self.refined_class_names = [
            " ".join(segment(class_name)) for class_name in self.classes
        ]

        print(f"refined_class_names: {self.refined_class_names}")

    def __getitem__(self, index):
        video_index = index % self.video_num
        path = self.video_lists[video_index]
        class_name = path.split("/")[-2]
        class_index = self.class_to_idx[class_name]

        vframes, aframes, info = torchvision.io.read_video(
            filename=path, pts_unit="sec", output_format="TCHW"
        )
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)

        if (end_frame_ind - start_frame_ind) < self.target_video_len:
            print(
                f"Error!!!! rank:{dist.get_rank()}; index:{index}; index%self.video_num:{index%self.video_num}; total frames: {total_frames}, start_frame_ind: {start_frame_ind}, end_frame_ind: {end_frame_ind}, file path: {path}; workaround with index+1"
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
        video = vframes[frame_indice]

        # videotransformer data proprecess
        video = self.transform(video)  # T C H W
        images = []
        image_names = []

        for i in range(self.use_image_num):
            while True:
                try:
                    video_frame_path = self.video_frame_files[index + i]
                    # HandstandPushupsv remove the last "v"
                    image_class_name = video_frame_path.split("_")[0][:-1]
                    image_class_index = self.class_to_idx[image_class_name]

                    refined_video_frame_path = "v_" + "_".join(
                        video_frame_path.split("_")[1:]
                    )

                    video_frame_path = os.path.join(
                        self.frame_data_path, image_class_name, refined_video_frame_path
                    )

                    image = Image.open(video_frame_path).convert("RGB")
                    image = self.image_tranform(image).unsqueeze(0)
                    images.append(image)
                    image_names.append(str(image_class_index))
                    break
                except Exception as e:
                    traceback.print_exc()
                    index = random.randint(0, self.video_frame_num - self.use_image_num)

        if self.use_image_num == 0:
            return {
                "video": video,
                "video_name": class_index,
                "video_text_prompt": self.refined_class_names[class_index],
                "image_name": "",
            }

        images = torch.cat(images, dim=0)
        assert len(images) == self.use_image_num
        assert len(image_names) == self.use_image_num

        image_names = "=====".join(image_names)

        video_cat = torch.cat([video, images], dim=0)

        # print(f"video_cat: {video_cat.shape} video_name: {class_index} image_names: {image_names}")

        return {
            "video": video_cat,
            "video_name": class_index,
            "image_name": image_names,
        }

    def __len__(self):
        return self.video_frame_num


if __name__ == "__main__":
    import argparse

    import torch.utils.data as Data
    import torchvision.transforms as transforms
    import video_transforms
    from PIL import Image

    # vframes, aframes, info = torchvision.io.read_video(filename="/mnt/xtan-jfs/UCF101/UCF-101/Shotput/v_Shotput_g16_c04.avi", pts_unit='sec', output_format='TCHW')
    # print(f"vframes: {vframes.shape}, aframes: {aframes.shape}, info: {info}")
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--use-image-num", type=int, default=0)
    parser.add_argument(
        "--data-path", type=str, default="/mnt/xtan-jfs/UCF101/UCF-101/"
    )
    parser.add_argument(
        "--frame-data-path", type=str, default="/mnt/xtan-jfs/UCF101/UCF-101-Image/"
    )
    parser.add_argument(
        "--frame-data-txt",
        type=str,
        default="/mnt/xtan-jfs/UCF101/UCF-101-train_256_list.txt",
    )
    config = parser.parse_args()

    temporal_sample = video_transforms.TemporalRandomCrop(
        config.num_frames * config.frame_interval
    )

    transform_ucf101 = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(256),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )

    ffs_dataset = UCF101Images(config, transform=transform_ucf101)
    ffs_dataloader = Data.DataLoader(
        dataset=ffs_dataset, batch_size=6, shuffle=True, num_workers=1
    )

    class_to_idx = ffs_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # for i, video_data in enumerate(ffs_dataloader):
    for video_data in ffs_dataloader:
        # print(type(video_data))
        video = video_data["video"]
        # video_name = video_data['video_name']
        print(video.shape)
        print(video_data["image_name"])
        image_name = video_data["image_name"]
        image_names = []
        for caption in image_name:
            single_caption = [int(item) for item in caption.split("=====")]
            image_names.append(torch.as_tensor(single_caption))
        print(image_names)

        # save the video to mp4
        video = rearrange(video, "b t c h w -> b t h w c")

        video = (
            ((video * 0.5 + 0.5) * 255)
            .add_(0.5)
            .clamp_(0, 255)
            .to(dtype=torch.uint8)
            .cpu()
        )

        import imageio

        def save_video(video, output_path):
            with imageio.get_writer(output_path, fps=8) as writer:
                for frame in video:
                    writer.append_data(frame)

        # Example usage

        # output_video_path = "output_video.mp4"
        # save_video(video[4,:16,...].numpy().astype(np.uint8), output_video_path)

        # print(f"video class: {idx_to_class[video_data['video_name'][4].item()]}")

        # #save a image
        # image = video[3,18,...]
        # image = Image.fromarray(np.uint8(image))
        # image.save('output_image.jpg')
        # print(f"image class: {idx_to_class[image_names[3][2].item()]}")
        break

        # print(video_name)
        # print(video_data[2])

        # for i in range(16):
        #     img0 = rearrange(video_data[0][0][i], 'c h w -> h w c')
        #     print('Label: {}'.format(video_data[1]))
        #     print(img0.shape)
        #     img0 = Image.fromarray(np.uint8(img0 * 255))
        #     img0.save('./img{}.jpg'.format(i))
