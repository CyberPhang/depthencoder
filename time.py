import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import time
from transformers import SiglipVisionModel
from data.load import load_data
from models.autoencoder import Autoencoder
from models.decoder import DepthDecoder
from utils import depth_to_viridis_gpu, VIRIDIS_LUT, to_grid, patchify, normalize_patches
from torch.nn.utils import clip_grad_norm_

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # model = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-224")
    # model = model.to(device)

    # batch_sizes = [1, 4, 8, 16, 32]
    # n_runs = 20

    # with torch.no_grad():
    #     for bs in batch_sizes:
    #         dummy = torch.randn(bs, 3, 224, 224, device=device)
            
    #         # warmup
    #         for _ in range(3):
    #             out = model(pixel_values=dummy, output_hidden_states=True).hidden_states
    #             # loss = out.last_hidden_state.mean()
    #             # loss.backward()

    #         torch.cuda.synchronize()
    #         start = time.perf_counter()
    #         for _ in range(n_runs):
    #             out = model(pixel_values=dummy, output_hidden_states=True).hidden_states
    #             # loss = out.last_hidden_state.mean()
    #             # loss.backward()
    #         torch.cuda.synchronize()
    #         elapsed = time.perf_counter() - start

    #         per_batch = elapsed / n_runs
    #         per_image = per_batch / bs
    #         print(f"batch_size={bs:3d} | {per_batch:.3f}s/batch | {per_image:.3f}s/image")


    train, dev, test = load_data(
        "./images/pixmo_images_224",
        "./images/pixmo_depth_224",
        batch_size=32,
    )

    # t0 = time.perf_counter()
    # for i, batch in enumerate(train):
    #     pass
    # print(f"total: {time.perf_counter() - t0:.3f}s")

    model = Autoencoder()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    # )

    for i, (rgb, depth) in enumerate(train):
        if i >= 20:
            break
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        rgb, depth = rgb.to(device), depth.to(device)
        optimizer.zero_grad()
        rgb = (rgb - 0.5) / 0.5
        depth_v = depth_to_viridis_gpu(depth, VIRIDIS_LUT)
        depth_v = (depth_v - 0.5) / 0.5
        torch.cuda.synchronize()
        print(f"basic preprocessing: {time.perf_counter() - t0:.3f}s")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        patches = patchify(depth)
        normed_patches, mean, std = normalize_patches(patches)
        torch.cuda.synchronize()
        print(f"patch + norm: {time.perf_counter() - t0:.3f}s")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_normed = model(rgb, depth_v)
        torch.cuda.synchronize()
        print(f"fwd: {time.perf_counter() - t0:.3f}s")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        l = F.l1_loss(pred_normed, normed_patches)
        torch.cuda.synchronize()
        print(f"loss: {time.perf_counter() - t0:.3f}s")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        l.backward()
        torch.cuda.synchronize()
        print(f"backward: {time.perf_counter() - t0:.3f}s")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.synchronize()
        print(f"clip grad: {time.perf_counter() - t0:.3f}s")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        print(f"optimizer step: {time.perf_counter() - t0:.3f}s")


    # print("Num of trainable params:")
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
