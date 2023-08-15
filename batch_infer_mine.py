from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import os
import requests
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ram import get_transform
from ram.models import ram, tag2text
from ram.utils import build_openset_label_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True)
    parser.add_argument("--backbone",
                        type=str,
                        choices=("swin_l", "swin_b"),
                        default=None,
                        help="If `None`, will judge from `--model-type`")
    parser.add_argument("--open-set",
                        type=bool,
                        default=False,
                        help=(
                            "Treat all categories in the taglist file as "
                            "unseen and perform open-set classification. Only "
                            "works with RAM."
                        ))
    # data
    parser.add_argument("--record-path",
                        type=str,
                        required=True)
    parser.add_argument("--input-size",
                        type=int,
                        default=384)
    parser.add_argument("--save-tags",
                        type=bool,
                        default=True,
                        help="save RAM tags to database.")
    # threshold
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold",
                       type=float,
                       default=None,
                       help=(
                           "Use custom threshold for all classes. Mutually "
                           "exclusive with `--threshold-file`. If both "
                           "`--threshold` and `--threshold-file` is `None`, "
                           "will use a default threshold setting."
                       ))
    group.add_argument("--threshold-file",
                       type=str,
                       default=None,
                       help=(
                           "Use custom class-wise thresholds by providing a "
                           "text file. Each line is a float-type threshold, "
                           "following the order of the tags in taglist file. "
                           "See `ram/data/ram_tag_list_threshold.txt` as an "
                           "example. Mutually exclusive with `--threshold`. "
                           "If both `--threshold` and `--threshold-file` is "
                           "`None`, will use default threshold setting."
                       ))
    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    # post process and validity check
    if args.backbone is None:
        args.backbone = "swin_l"

    return args


def load_dataset(
    img_path: str,
    input_size: int,
    batch_size: int,
    num_workers: int
) -> Tuple[DataLoader, Dict]:
    imglist = []
    for filename in os.listdir(img_path):
        filepath = os.path.join(img_path, filename)
        imglist.append(filepath)

    class _Dataset(Dataset):
        def __init__(self):
            self.transform = get_transform(input_size)

        def __len__(self):
            return len(imglist)

        def __getitem__(self, index):
            try:
                img = Image.open(imglist[index])
            except (OSError, FileNotFoundError, UnidentifiedImageError):
                img = Image.new('RGB', (10, 10), 0)
                print("Error loading image:", imglist[index])
            return self.transform(img)

    loader = DataLoader(
        dataset=_Dataset(),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    info = {
        # "taglist": taglist,
        "imglist": imglist,
        # "annot_file": annot_file,
        "img_path": img_path
    }
    return loader, info

def get_tag_list(
    open_set: bool
) -> Optional[List[str]]:
    """Get indices of required categories in the label system."""
    if not open_set:
        model_taglist_file = "ram/data/ram_tag_list.txt"
        with open(model_taglist_file, "r", encoding="utf-8") as f:
            model_taglist = [line.strip() for line in f]
            return model_taglist
    else:
        return None


def load_thresholds(
    threshold: Optional[float],
    threshold_file: Optional[str],
    open_set: bool,
    class_idxs: List[int],
    num_classes: int,
) -> List[float]:
    """Decide what threshold(s) to use."""
    if not threshold_file and not threshold:  # use default
        if not open_set:  # use class-wise tuned thresholds
            ram_threshold_file = "ram/data/ram_tag_list_threshold.txt"
            with open(ram_threshold_file, "r", encoding="utf-8") as f:
                idx2thre = {
                    idx: float(line.strip()) for idx, line in enumerate(f)
                }
                return [idx2thre[idx] for idx in class_idxs]
        else:
            return [0.5] * num_classes
    elif threshold_file:
        with open(threshold_file, "r", encoding="utf-8") as f:
            thresholds = [float(line.strip()) for line in f]
        assert len(thresholds) == num_classes
        return thresholds
    else:
        return [threshold] * num_classes


def gen_pred_file(
    imglist: List[str],
    tags: List[List[str]],
    img_root: str,
    pred_file: str
) -> None:
    """Generate text file of tag prediction results."""
    with open(pred_file, "w", encoding="utf-8") as f:
        for image, tag in zip(imglist, tags):
            # should be relative to img_root to match the gt file.
            s = str(Path(image).relative_to(img_root))
            if tag:
                s = s + "," + ",".join(tag)
            f.write(s + "\n")


def load_ram(
    backbone: str,
    checkpoint: str,
    input_size: int,
    taglist: List[str],
    open_set: bool,
    class_idxs: List[int],
) -> Module:
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)
    # trim taglist for faster inference
    if open_set:
        print("Building tag embeddings ...")
        label_embed, _ = build_openset_label_embedding(taglist)
        model.label_embed = Parameter(label_embed.float())
    else:
        model.label_embed = Parameter(model.label_embed[class_idxs, :])
    return model.to(device).eval()


@torch.no_grad()
def forward_ram(model: Module, imgs: Tensor) -> Tensor:
    image_embeds = model.image_proj(model.visual_encoder(imgs.to(device)))
    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(device)
    label_embed = relu(model.wordvec_proj(model.label_embed)).unsqueeze(0)\
        .repeat(imgs.shape[0], 1, 1)
    tagging_embed, _ = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )
    return sigmoid(model.fc(tagging_embed).squeeze(-1))


def print_write(f: TextIO, s: str):
    print(s)
    f.write(s + "\n")


def _save_tags(record_path: str, vehicle_id: str, tags: list):
    json_payload = {
        'record_path': record_path,
        'vehicle_id': vehicle_id,
        'frame_tags': tags
    }
    print(f'json_payload: {json_payload}')
    response = requests.post(
        url="http://ram-tag-index-service-dev.autra.tech/write",
        json=json_payload,
        timeout=3
    )
    json_obj = response.json()
    response.close()
    print(f'json_obj: {json_obj}')


def _generate_tags(
        img_list: List[str],
        tags: List[List[str]],
        img_root: str):
    img2time = {}
    with open(os.path.join(img_root, 'timestamps'), 'r', encoding="utf-8") as f:
        for line in f.readlines():
            tokens = line.split(' ')
            img2time[tokens[0]] = float(tokens[1])

    format_tags = []
    for img_path, tag in zip(img_list, tags):
        # should be relative to img_root to match the gt file.
        paths = img_path.split('/')
        img_name = paths[-1].split('.')[0]
        if img_name in img2time:
            format_tags.append({'timestamp': img2time[img_name], 'tags': tag})
    return format_tags


def _format_img_path(record_path: str):
    img_path = os.path.join(record_path, '_apollo_sensor_camera_upmiddle_right_60h_image_compressed')
    record_name = record_path.split('/')[-1]
    vehicle_id = record_name.split('_')[0]
    return vehicle_id, img_path


if __name__ == "__main__":
    args = parse_args()

    # set up output paths
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred_file, pr_file, ap_file, summary_file, logit_file = [
        output_dir + "/" + name for name in
        ("pred.txt", "pr.txt", "ap.txt", "summary.txt", "logits.pth")
    ]
    with open(summary_file, "w", encoding="utf-8") as f:
        print_write(f, "****************")
        for key in (
            "backbone", "checkpoint", "open_set",
            "record_path", "input_size",
            "threshold", "threshold_file",
            "output_dir", "batch_size", "num_workers"
        ):
            print_write(f, f"{key}: {getattr(args, key)}")
        print_write(f, "****************")



    # prepare data
    vehicle_id, img_path = _format_img_path(args.record_path)
    loader, info = load_dataset(
        img_path=img_path,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    imglist, img_path = info["imglist"], info["img_path"]

    # inference all tags
    tag_list = get_tag_list(args.open_set)
    class_idxs = range(len(tag_list))

    # set up threshold(s)
    thresholds = load_thresholds(
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        open_set=args.open_set,
        class_idxs=class_idxs,
        num_classes=len(class_idxs)
    )

    # inference
    if Path(logit_file).is_file():
        logits = torch.load(logit_file)
    else:
        # load model
        model = load_ram(
            backbone=args.backbone,
            checkpoint=args.checkpoint,
            input_size=args.input_size,
            taglist=tag_list,
            open_set=args.open_set,
            class_idxs=class_idxs
        )

        # inference
        logits = torch.empty(len(imglist), len(class_idxs))
        pos = 0
        for imgs in tqdm(loader, desc="inference"):
            out = forward_ram(model, imgs)
            bs = imgs.shape[0]
            logits[pos:pos+bs, :] = out.cpu()
            pos += bs

        # save logits, making threshold-tuning super fast
        torch.save(logits, logit_file)

    # filter with thresholds
    pred_tags = []
    for scores in logits.tolist():
        pred_tags.append([
            tag_list[i] for i, s in enumerate(scores) if s >= thresholds[i]
        ])

    # generate result file
    gen_pred_file(imglist, pred_tags, img_path, pred_file)
    if args.save_tags:
        format_tags = _generate_tags(imglist, pred_tags, img_path)
        _save_tags(args.record_path, vehicle_id, format_tags)
