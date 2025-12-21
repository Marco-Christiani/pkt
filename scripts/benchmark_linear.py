#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "torch",
# ]
# ///

import argparse
import time

import numpy as np
import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, x):
        return self.linear(x)


def make_test_data(m, n, batch, seed=42):
    rng = np.random.RandomState(seed)
    scale = np.sqrt(2.0 / (m + n))
    W = rng.normal(0.0, scale, size=(m, n)).astype(np.float32)
    x = rng.normal(0.0, 1.0, size=(batch, n)).astype(np.float32)
    noise = rng.normal(0.0, 0.01, size=(batch, m)).astype(np.float32)
    target = x @ W.T + noise
    return W, x, target


def train_pytorch(
    W_init,
    x,
    target,
    num_steps,
    lr,
    device="cuda",
    warmup_steps=2,
    use_compile=False,
):
    m, n = W_init.shape
    W = torch.from_numpy(W_init.copy()).to(device)
    x_t = torch.from_numpy(x).to(device)
    target_t = torch.from_numpy(target).to(device)

    model = LinearModel(n, m).to(device)
    model.linear.weight.data = W
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if use_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch build")
        model = torch.compile(model)

    loss_curve = []
    if device == "cuda":
        torch.cuda.synchronize()

    for _ in range(warmup_steps):
        y_pred = model(x_t)
        loss = criterion(y_pred, target_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_steps):
        y_pred = model(x_t)
        loss = criterion(y_pred, target_t)
        loss_curve.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "final_loss": loss_curve[-1],
        "loss_curve": loss_curve,
        "time_ms": elapsed_ms,
        "time_per_step_ms": elapsed_ms / num_steps,
    }


def parse_cuda_log(path):
    if not path:
        return None, None, None
    loss = time_total = time_step = None
    with open(path, "r") as f:
        for line in f:
            if "Final loss (host recompute):" in line:
                loss = float(line.split(":")[1].strip())
            elif line.startswith("Time:") and "ms" in line:
                time_total = float(line.split(":")[1].strip().replace("ms", ""))
            elif line.startswith("Time per step:"):
                time_step = float(line.split(":")[1].strip().replace("ms", ""))
    return loss, time_total, time_step


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs CUDA linear training")
    parser.add_argument("--cuda-log", type=str, help="Path to CUDA implementation log file")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--m", type=int, default=64, help="Output size")
    parser.add_argument("--n", type=int, default=128, help="Input size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps not timed")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for the model during training",
    )
    parser.add_argument("--export-data", type=str, help="Export test data to NPZ file")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for PyTorch run")
    args = parser.parse_args()

    m = args.m
    n = args.n
    lr = args.lr
    steps = args.steps
    batch = args.batch
    print("=" * 60)
    print(f"Model W:[{m},{n}] batch={batch} steps={steps} lr={lr}")
    print(f"Params: {m*n}")
    print("=" * 60)

    W_init, x, target = make_test_data(m, n, batch)
    y_init = x @ W_init.T
    init_loss = np.mean((y_init - target) ** 2)
    print(f"Initial loss: {init_loss:.6e}")

    torch_res = train_pytorch(
        W_init,
        x,
        target,
        steps,
        lr,
        device=args.device,
        warmup_steps=args.warmup_steps,
        use_compile=args.compile,
    )
    print(f"\nPyTorch {args.device}:")
    print(
        f"  loss={torch_res['final_loss']:.6e} time={torch_res['time_ms']:.3f} ms "
        f"({torch_res['time_per_step_ms']:.4f} ms/step)"
    )

    cuda_loss, cuda_time, cuda_time_step = parse_cuda_log(args.cuda_log)
    if cuda_loss is not None:
        print("\nCUDA impl (log):")
        print(f"  loss={cuda_loss:.6e} time={cuda_time:.3f} ms ({cuda_time_step:.4f} ms/step)")

    results = {
        "pytorch": torch_res,
        "cuda_impl": {"loss": cuda_loss, "time_ms": cuda_time},
        "test_data": {"W": W_init, "x": x, "target": target},
    }

    if args.export_data:
        np.savez(
            args.export_data,
            W=results["test_data"]["W"],
            x=results["test_data"]["x"],
            target=results["test_data"]["target"],
            pytorch_loss_curve=results["pytorch"]["loss_curve"],
        )
        print(f"\nSaved data to {args.export_data}")


if __name__ == "__main__":
    main()
