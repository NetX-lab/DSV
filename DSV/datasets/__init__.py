import random

import torch
import webdataset as wds
from DSV.datasets import video_transforms
from torchvision import transforms
from video2dataset.dataloader import get_video_dataset
from video2dataset.dataloader.transform import VideoResizer
from video2dataset.dataloader.video_decode import VideoDecorder
from webdataset import WebLoader

from DSV.datasets.openvid_datasets import OpenVidDataset
from DSV.datasets.ucf101_datasets import UCF101
from DSV.datasets.videogen_datasets import VideoGenDataset
from DSV.datasets.webvid_datasets import WebVid
from DSV.datasets.dummy_datasets import DummyDataset


def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(
        args.num_frames * args.frame_interval
    )  # 16 1

    if args.dataset == "ucf101":
        transform_ucf101 = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(args.image_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )
        return UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
   
    elif args.dataset == "dummy":
        return DummyDataset(args)

    elif args.dataset == "videogen":
        return VideoGenDataset(args)

    elif args.dataset == "openvid":
        return OpenVidDataset(args)

    elif args.dataset == "webvid":
        SHARDS = "/mnt/xtan-jfs/webvid-10M/dataset/{00000..00900}.tar"
        decoder_kwargs = {
            "n_frames": args.num_frames,  # get 8 frames from each video
            "fps": args.fps,  # downsample to 10 FPS
            "num_threads": 12,  # use 12 threads to decode the video
        }
        resize_size = crop_size = args.image_size
        batch_size = args.local_batch_size

        def video_transform(x):
            for key in x:
                if key in [
                    "mp4",
                    "ogv",
                    "mjpeg",
                    "avi",
                    "mov",
                    "h264",
                    "mpg",
                    "webm",
                    "wmv",
                ]:
                    x[key] = x[key].to(torch.uint8).float() / 255.0
                    # T,H,W,C
                    x[key] = x[key].flip(-2) if random.random() < 0.5 else x[key]
                    # T,C,H,W
                    x[key] = x[key].permute(0, 3, 1, 2)
                    x[key] = transforms.Normalize(
                        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False
                    )(x[key])
                    # x[key]=x[key].permute(0, 2, 3, 1) # T,H,W,C
            return x

        web_dataset = wds.WebDataset(
            SHARDS, nodesplitter=wds.split_by_node, resampled=True
        ).shuffle(1000)

        handler = wds.handlers.warn_and_continue

        transform_handler = wds.reraise_exception

        def reassemble(x):
            # Separate video data and metadata (e.g., resolution, frame rate, etc.)
            # {
            # 'mp4': (Video data, {'width': 1920, 'height': 1080, 'fps': 30}),
            # 'txt': 'Video description'
            # }
            #     {
            #     'mp4': video data,
            #     'width': 1920,
            #     'height': 1080,
            #     'fps': 30,
            #     'txt': 'video description'
            #   }

            new_dict = {}

            # make sure the mp4 and text are in the dict
            assert "mp4" in x and "txt" in x

            for key in x:
                if key not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
                    continue

                # this is updating the output of video decoders
                if isinstance(x[key], tuple) and len(x[key]) == 2:
                    new_dict.update(
                        {f"{subk}": x[key][-1][subk] for subk in x[key][-1]}
                    )

                x[key] = x[key][0]
            x.update(new_dict)
            del new_dict
            return x

        web_dataset = (
            web_dataset.decode(
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

        web_dataset = web_dataset.batched(args.local_batch_size)

        return web_dataset

    else:
        raise NotImplementedError(args.dataset)
