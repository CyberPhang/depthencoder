from pathlib import Path
from PIL import Image
from transformers import pipeline
from tqdm import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
pipe = pipeline(
    task="depth-estimation", 
    model="depth-anything/Depth-Anything-V2-Large-hf",
    device=device,
    torch_dtype=torch.float16
)

output_dir = Path("../images/pixmo_depth_224")
output_dir.mkdir(exist_ok=True)
image_dir = Path("../images/pixmo_images_224")

for img_path in tqdm(list(image_dir.glob("**/*.jpg"))):
    out_path = output_dir / img_path.name
    if out_path.exists():
        continue
    img = Image.open(img_path).convert("RGB")
    depth = pipe(img)["depth"]
    depth.save(out_path)