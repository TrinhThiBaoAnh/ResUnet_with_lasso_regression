import warnings

warnings.simplefilter("ignore", UserWarning)

import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import prune

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from resunet import ResUnet
from resunetplusplus import ResUnetPlusPlus
from utils import log_images, dsc, create_classification_report
from prune import measure_global_sparsity


def train_model(model, loaders, args, device):
    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.NAdam(model.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0
    l1_regularization_strength = 1e-3
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = model(x)

                    loss = dsc_loss(y_pred, y_true)
                    l1_reg = torch.tensor(0.).to(device)
                    for module in model.modules():
                        mask = None
                        weight = None
                        for name, buffer in module.named_buffers():
                            if name == "weight_mask":
                                mask = buffer
                        for name, param in module.named_parameters():
                            if name == "weight_orig":
                                weight = param
                        # We usually only want to introduce sparsity to weights and prune weights.
                        # Do the same for bias if necessary.
                        if mask is not None and weight is not None:
                            l1_reg += torch.norm(mask * weight, 1)

                    loss += l1_regularization_strength * l1_reg

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])])
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])])
                        if (epoch % args.vis_freq
                                == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_dsc = np.mean(dsc(
                    validation_pred,
                    validation_true,
                ))
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.weights,"unet-{0}.pt".format(best_validation_dsc)))
                loss_valid = []
    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    model = ResUnetPlusPlus(in_channels=Dataset.in_channels,
                 out_channels=Dataset.out_channels)
    if args.pretrained:
        model.load_state_dict(
            torch.load(args.pretrained, map_location=device))
    model.to(device)

    for i in range(args.prune_iters):
        if args.grouped_pruning == True:
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=args.conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=args.conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=args.linear_prune_amount)
        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print('Model sparsity after {0} iter: {1}'.format(i, sparsity))
        train_model(model, loaders, args, device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(
        model,
        weight=True,
        bias=False,
        conv2d_use_mask=True,
        linear_use_mask=False)
    print('Model sparsity at last: {0}'.format(sparsity))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale,
                             angle=args.aug_angle,
                             flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="test",
        image_size=args.image_size,
        random_sampling=False,
    )
    print("Training: ", len(train))
    print("Validation: ", len(valid))
    return train, valid


# def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
#     dsc_list = []
#     num_slices = np.bincount([p[0] for p in patient_slice_index])
#     index = 0
#     for p in range(len(num_slices)):
#         y_pred = np.array(validation_pred[index : index + num_slices[p]])
#         y_true = np.array(validation_true[index : index + num_slices[p]])
#         dsc_list.append(dsc(y_pred, y_true))
#         index += num_slices[p]
#     return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training ResUNetPlusPlus model for segmentation of brain MRI")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help=
        "number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument("--weights",
                        type=str,
                        default="./weights",
                        help="folder to save weights")
    parser.add_argument("--logs",
                        type=str,
                        default="./logs",
                        help="folder to save logs")
    parser.add_argument("--images",
                        type=str,
                        default="/home/mcn/ba/UTThucquan",
                        help="root folder with images")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument('--pretrained',
                        type=str,
                        default=None,
                        help='Path to pretrained ')

    parser.add_argument("--prune-iters",
                        type=int,
                        default=2,
                        help="number of iteration for pruning")
    parser.add_argument('--grouped_pruning', type=bool, default=True)
    parser.add_argument('--conv2d_prune_amount', type=float, default=0.4)
    parser.add_argument('--linear_prune_amount', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
