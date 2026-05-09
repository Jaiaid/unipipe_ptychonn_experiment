import argparse
import os
import sys
# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add it to the system path
sys.path.append(parent_dir)


import h5py

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path
from ptychonn.parameters import *

H, W = 64, 64
NCONV = 32


def get_dataset(is_large_dataset:bool=False, linecount: int = DIFFRLINE_L) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if is_large_dataset:
        diffr_data = h5py.File(os.path.join(LARGE_DATASET_DIR, LARGE_DATASET_FILE))["data"]["reciprocal"][:]
        diffr_data = diffr_data.reshape(DIFFRLINE_L, SCANPOINT_L, H_L, W_L)[:linecount]
        ground_truth_data = h5py.File(os.path.join(LARGE_DATASET_DIR, LARGE_DATASET_FILE))["data"]["real"][:linecount*SCANPOINT_L]

        print(f"Loaded dataset with diffr_data shape: {diffr_data.shape} | ground_truth_data shape: {ground_truth_data.shape}")
        diffr_data_red = np.zeros((diffr_data.shape[0],diffr_data.shape[1],H,W), float)
        for i in range(diffr_data.shape[0]):
            for j in range(diffr_data.shape[1]):
                diffr_data_red[i,j] = resize(diffr_data[i,j],(H,W),preserve_range=True, anti_aliasing=True)
                diffr_data_red[i,j] = np.where(diffr_data_red[i,j]<3,0,diffr_data_red[i,j])

        ground_truth_amp_tmp = np.abs(ground_truth_data)
        ground_truth_ph_tmp = np.angle(ground_truth_data)

        ground_truth_amp = np.zeros((ground_truth_amp_tmp.shape[0], H, W), float)
        ground_truth_ph = np.zeros((ground_truth_ph_tmp.shape[0], H, W), float)
        for i in range(ground_truth_data.shape[0]):
            # for NN feeding
            ground_truth_amp[i] = resize(ground_truth_amp_tmp[i],(64,64), preserve_range=True, anti_aliasing=True)
            ground_truth_ph[i] = resize(ground_truth_ph_tmp[i],(64,64), preserve_range=True, anti_aliasing=True)
    else:
        diffr_data = np.load(DATA_DIFFR_PATH)["arr_0"][:linecount]
        ground_truth_data = np.load(REAL_SPACE_PATH)[:linecount*DIFFRLINE]
        
        print(f"Loaded dataset with diffr_data shape: {diffr_data.shape} | ground_truth_data shape: {ground_truth_data.shape}")
        diffr_data_red = np.zeros((diffr_data.shape[0],diffr_data.shape[1],H,W), float)
        for i in range(diffr_data.shape[0]):
            for j in range(diffr_data.shape[1]):
                diffr_data_red[i,j] = resize(diffr_data[i,j,32:-32,32:-32],(H,W),preserve_range=True, anti_aliasing=True)
                diffr_data_red[i,j] = np.where(diffr_data_red[i,j]<3,0,diffr_data_red[i,j])

        ground_truth_amp = np.abs(ground_truth_data)
        ground_truth_ph = np.angle(ground_truth_data)
    
    diffr_data_red = diffr_data_red.reshape(-1, H, W)[:, np.newaxis, :, :]

    print(f"Processed ground truth data into amplitude and phase with shapes: {ground_truth_amp.shape} | {ground_truth_ph.shape}")

    total_image_count = diffr_data_red.shape[0]
    valid_image_count = int(total_image_count * 0.3)
    train_image_count = total_image_count - valid_image_count
    print(f"Total images: {total_image_count} | Train: {train_image_count} | Valid: {valid_image_count}")

    X_train = diffr_data_red[:train_image_count].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_I_train = ground_truth_amp[:train_image_count].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_phi_train = ground_truth_ph[:train_image_count].reshape(-1,H,W)[:,np.newaxis,:,:]
    print(f"Train set: {X_train.shape} | {Y_I_train.shape} | {Y_phi_train.shape}")

    X_valid = diffr_data_red[train_image_count:].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_I_valid = ground_truth_amp[train_image_count:].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_phi_valid = ground_truth_ph[train_image_count:].reshape(-1,H,W)[:,np.newaxis,:,:]
    print(f"Valid set: {X_valid.shape} | {Y_I_valid.shape} | {Y_phi_valid.shape}")

    return {"train": (X_train, Y_I_train, Y_phi_train),
            "valid": (X_valid, Y_I_valid, Y_phi_valid)}

class ReconModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=NCONV, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV, NCONV, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(NCONV, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(NCONV * 2, NCONV * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 4, NCONV * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(NCONV * 4, NCONV * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 4, NCONV * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(NCONV * 4, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(NCONV * 2, 1, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(NCONV * 4, NCONV * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 4, NCONV * 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(NCONV * 4, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NCONV * 2, NCONV * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(NCONV * 2, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)
        ph = ph * np.pi
        return amp, ph



def build_dataloaders(
    is_large_dataset: bool,
    linecount: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = get_dataset(
        is_large_dataset=is_large_dataset, linecount=linecount)
    x_train, y_amp_train, y_ph_train = dataset["train"]
    x_valid, y_amp_valid, y_ph_valid = dataset["valid"]

    train_data = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_amp_train, dtype=torch.float32),
        torch.tensor(y_ph_train, dtype=torch.float32),
    )
    valid_data = TensorDataset(
        torch.tensor(x_valid, dtype=torch.float32),
        torch.tensor(y_amp_valid, dtype=torch.float32),
        torch.tensor(y_ph_valid, dtype=torch.float32),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(f"Train: {len(train_data)} | Valid: {len(valid_data)}")
    print(f"X train: {x_train.shape} | X valid: {x_valid.shape}")

    return train_loader, valid_loader


def update_saved_model(model: nn.Module, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # for f in out_dir.iterdir():
    #     if f.is_file():
    #         f.unlink()

    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), out_dir / "best_model.pth")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CyclicLR,
) -> tuple[float, float, float, list[float]]:
    model.train()
    total_loss = 0.0
    amp_loss = 0.0
    ph_loss = 0.0
    lr_trace: list[float] = []

    for ft_images, amps, phs in tqdm(loader, desc="Train", leave=False):
        ft_images = ft_images.to(device)
        amps = amps.to(device)
        phs = phs.to(device)

        pred_amps, pred_phs = model(ft_images)
        loss_a = criterion(pred_amps, amps)
        loss_p = criterion(pred_phs, phs)
        loss = loss_a + loss_p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.detach().item()
        amp_loss += loss_a.detach().item()
        ph_loss += loss_p.detach().item()
        lr_trace.append(scheduler.get_last_lr()[0])

    nbatches = max(1, len(loader))
    return total_loss / nbatches, amp_loss / nbatches, ph_loss / nbatches, lr_trace


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    amp_loss = 0.0
    ph_loss = 0.0

    for ft_images, amps, phs in tqdm(loader, desc="Valid", leave=False):
        ft_images = ft_images.to(device)
        amps = amps.to(device)
        phs = phs.to(device)

        pred_amps, pred_phs = model(ft_images)
        loss_a = criterion(pred_amps, amps)
        loss_p = criterion(pred_phs, phs)
        loss = loss_a + loss_p

        total_loss += loss.detach().item()
        amp_loss += loss_a.detach().item()
        ph_loss += loss_p.detach().item()

    nbatches = max(1, len(loader))
    return total_loss / nbatches, amp_loss / nbatches, ph_loss / nbatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN encoder-decoder ptychography model")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--linecount", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-per-gpu", type=int, default=64)
    parser.add_argument("--lr-per-gpu", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-type", type=str, default="1.25M", choices=["1.25M", "5M", "10M", "20M"])
    parser.add_argument("--large-dataset", action="store_true", help="Use the larger dataset for training")
    parser.add_argument("--model-dir", type=str, help="model directory where the trained model will be saved (overrides default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_save_path = Path(args.model_dir)

    n_gpus = max(1, torch.cuda.device_count())
    batch_size = n_gpus * args.batch_per_gpu
    lr = n_gpus * args.lr_per_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device} | GPUs detected: {torch.cuda.device_count()}")
    print(f"Batch size: {batch_size} | Learning rate: {lr}")

    train_loader, valid_loader = build_dataloaders(
        linecount=args.linecount, is_large_dataset=args.large_dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    model = ReconModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iterations_per_epoch = int(np.floor((len(train_loader.dataset)) / batch_size) + 1)
    step_size = 6 * iterations_per_epoch
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr / 10,
        max_lr=lr,
        step_size_up=step_size,
        cycle_momentum=False,
        mode="triangular2",
    )

    print(f"CyclicLR step_size_up: {step_size} ({step_size / max(1, iterations_per_epoch):.2f} epochs)")

    best_val = float("inf")
    metrics = {"losses": [], "val_losses": [], "lrs": []}

    for epoch in range(args.epochs):
        tr_total, tr_amp, tr_ph, lrs = train_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        va_total, va_amp, va_ph = validate_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            criterion=criterion,
        )

        metrics["losses"].append([tr_total, tr_amp, tr_ph])
        metrics["val_losses"].append([va_total, va_amp, va_ph])
        metrics["lrs"].extend(lrs)

        print(f"Epoch {epoch:03d} | FT  train={tr_total:.5f} val={va_total:.5f}")
        print(f"Epoch {epoch:03d} | Amp train={tr_amp:.5f} val={va_amp:.5f}")
        print(f"Epoch {epoch:03d} | Ph  train={tr_ph:.5f} val={va_ph:.5f}")
        if lrs:
            print(f"Epoch {epoch:03d} | LR end={lrs[-1]:.7f}")

        if va_total < best_val:
            print(f"Saving improved model: {best_val:.5f} -> {va_total:.5f}")
            best_val = va_total
            update_saved_model(model, model_save_path)

    np.save(model_save_path / "metrics_losses.npy", np.array(metrics["losses"], dtype=np.float32))
    np.save(model_save_path / "metrics_val_losses.npy", np.array(metrics["val_losses"], dtype=np.float32))
    np.save(model_save_path / "metrics_lrs.npy", np.array(metrics["lrs"], dtype=np.float32))
    print(f"Training finished. Artifacts saved in: {model_save_path}")


if __name__ == "__main__":
    main()