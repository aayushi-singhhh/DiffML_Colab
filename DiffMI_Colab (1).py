# Generated from: DiffMI_Colab (1).ipynb
# Converted at: 2026-06-12T16:23:21.909Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # DiffMI — Training-Free Model Inversion Attack
# ### Run cells top to bottom. Do NOT skip any cell.
# **First: Runtime → Change runtime type → T4 GPU**


# ═══════════════════════════════════════════════
# CELL 1 — Install packages (run once per session)
# ═══════════════════════════════════════════════
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q', '--no-deps'])

# Install only what is missing, no version conflicts
!pip install diffusers==0.30.0 -q --no-deps
!pip install accelerate==0.27.2 -q --no-deps
!pip install transformers==4.41.0 -q --no-deps
!pip install facenet-pytorch -q --no-deps
!pip install huggingface_hub -q
!pip install timm -q


print('Packages installed')

!pip install -q tokenizers -U
!pip install -q transformers -U
!pip install -q diffusers -U
!pip install -q accelerate -U
!pip install -q facenet-pytorch --no-deps
!pip install -q timm
print("Done — Restart Session now")

!pip install torch torchvision --upgrade -q
print("Done — Runtime → Restart Session NOW")
import tokenizers, transformers, diffusers, torch
print(tokenizers.__version__)
print(transformers.__version__)
print(diffusers.__version__)
print(torch.__version__)
print(torch.cuda.is_available())

import tokenizers, transformers, diffusers, torch
print(tokenizers.__version__)
print(transformers.__version__)
print(diffusers.__version__)
print(torch.__version__)
print(torch.cuda.is_available())

# ═══════════════════════════════════════════════
# CELL 2 — Imports and device check
# ═══════════════════════════════════════════════
import torch
import torch.nn.functional as F
import numpy as np
import os, gc, glob, random, warnings
from scipy import stats
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch : {torch.__version__}')
print(f'Device  : {device}')
if device.type == 'cuda':
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU')

# ═══════════════════════════════════════════════
# CELL 3 — Load DDPM (downloads ~450MB once)
# ═══════════════════════════════════════════════
from diffusers import UNet2DModel, DDPMScheduler

print('Downloading DDPM pretrained on CelebA-HQ 256x256 ...')
unet = UNet2DModel.from_pretrained(
    'google/ddpm-celebahq-256',
    use_safetensors=False
).to(device).eval()

scheduler = DDPMScheduler.from_pretrained('google/ddpm-celebahq-256')
print('DDPM ready')

# ═══════════════════════════════════════════════
# CELL 4 — Load ArcFace + MTCNN
# ═══════════════════════════════════════════════
from facenet_pytorch import InceptionResnetV1, MTCNN

arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn   = MTCNN(image_size=160, margin=32, keep_all=False, device=device)
print('ArcFace + MTCNN ready')

# ═══════════════════════════════════════════════
# CELL 5 — Core helper functions
# ═══════════════════════════════════════════════

def free_mem():
    gc.collect()
    torch.cuda.empty_cache()


def generate_image(latent, steps=20):
    """Run DDPM denoising: latent (1,3,256,256) -> image (1,3,256,256) in [-1,1]"""
    scheduler.set_timesteps(steps)
    x = latent.clone().to(device)
    with torch.no_grad():
        for t in scheduler.timesteps:
            out   = unet(x, t)
            x     = scheduler.step(out.sample, t, x).prev_sample
    return x


def to_pil(tensor):
    """Convert (1,3,H,W) tensor [-1,1] to PIL Image."""
    img = (tensor.squeeze().cpu().clamp(-1, 1) + 1) / 2   # [0,1]
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(img)


def get_face_embedding(inp):
    """
    inp: tensor (1,3,H,W) in [-1,1]  OR  PIL Image
    Returns (embedding (1,512), detection_confidence)
    embedding is None if no face detected.
    """
    pil = to_pil(inp) if torch.is_tensor(inp) else inp
    try:
        face, prob = mtcnn(pil, return_prob=True)
    except Exception:
        return None, 0.0
    if face is None or prob is None:
        return None, 0.0
    with torch.no_grad():
        emb = arcface(face.unsqueeze(0).to(device))
        emb = F.normalize(emb, p=2, dim=1)
    return emb, float(prob)


def cosine_sim(a, b):
    return F.cosine_similarity(a, b, dim=1).item()


print('Helper functions defined')

# ═══════════════════════════════════════════════
# CELL 6 — Sanity check: generate one face
# ═══════════════════════════════════════════════
free_mem()
test_latent = torch.randn(1, 3, 256, 256).to(device)
test_img    = generate_image(test_latent, steps=20)
test_pil    = to_pil(test_img)

emb, conf = get_face_embedding(test_img)
print(f'Face detected : {emb is not None}')
print(f'Confidence    : {conf:.4f}')
if emb is not None:
    print(f'Embedding dim : {emb.shape}')   # expect (1, 512)
display(test_pil)
print('Sanity check passed — pipeline works end to end')

# ═══════════════════════════════════════════════
# CELL 7 — Step (a): Build latent code pool
# Paper Section III-D
# Filters random Gaussian codes by:
#   1. D'Agostino K2 normality test
#   2. MTCNN face detection
# Run once — saves to pool.pt and reloads next time.
# ═══════════════════════════════════════════════

def build_pool(pool_size=100, tau_k=0.05, tau_d=0.80,
               steps=20, path='pool.pt'):

    if os.path.exists(path):
        data = torch.load(path, weights_only=False)
        print(f'Loaded existing pool: {len(data)} codes')
        return data

    pool, tries = [], 0
    bar = tqdm(total=pool_size, desc='Building pool')

    while len(pool) < pool_size:
        tries += 1
        free_mem()

        # Sample random Gaussian latent
        xG = torch.randn(1, 3, 256, 256).to(device)

        # --- Stage 1: Normality test (fast) ---
        flat = xG.flatten().cpu().numpy()
        samp = flat[np.random.choice(len(flat), 5000, replace=False)]
        _, p_k = stats.normaltest(samp)
        if p_k < tau_k:
            continue

        # --- Stage 2: Face detection (slower) ---
        x_hat = generate_image(xG, steps=steps)
        _, p_d = get_face_embedding(x_hat)
        if p_d < tau_d:
            continue

        pool.append((xG.cpu(), x_hat.cpu()))
        bar.update(1)

    bar.close()
    torch.save(pool, path)
    print(f'Pool saved: {len(pool)} codes in {tries} attempts '
          f'({len(pool)/tries*100:.1f}% acceptance rate)')
    return pool


# Build pool — ~5 mins on T4 with these settings
pool = build_pool(pool_size=100, tau_k=0.05, tau_d=0.80,
                  steps=20, path='pool.pt')

# ═══════════════════════════════════════════════
# CELL 8 — Step (b): Top-N selection
# Paper Section III-E
# Rank pool codes by similarity to target embedding.
# ═══════════════════════════════════════════════

def select_top_n(pool, target_emb, N=3):
    scored = []
    for xG, x_hat in tqdm(pool, desc='Ranking pool'):
        emb, _ = get_face_embedding(x_hat.to(device))
        sim     = cosine_sim(emb, target_emb) if emb is not None else -1.0
        scored.append((xG, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:N]
    print(f'Top-{N} initial similarities: {[round(s,4) for _,s in top]}')
    return top


print('select_top_n defined')

# ═══════════════════════════════════════════════
# CELL 9 — Step (c): Ranked Adversary attack
# Paper Section III-F
# Uses finite-difference gradient estimation so
# DDPM never needs to store gradients -> no OOM.
# ═══════════════════════════════════════════════

def attack_one(xG_init, target_emb,
               epsilon=35.0,   # L2 perturbation budget
               tau_c=0.85,     # confidence threshold for early stop
               t_max=30,       # max iterations
               steps=15,       # DDPM denoising steps per call
               n_fd=8,         # finite-difference directions
               lr=2.0,         # step size
               verbose=True):
    """
    Optimize perturbation delta on xG_init so that
    generate_image(xG_init + delta) matches target_emb.
    """
    free_mem()
    xG    = xG_init.clone().to(device)
    delta = torch.zeros_like(xG)
    best_sim, best_img = -1.0, None

    for i in range(t_max):
        free_mem()

        # ---- Generate current image (no grad) ----
        with torch.no_grad():
            x_hat = generate_image((xG + delta).clamp(-3, 3), steps=steps)

        emb, _ = get_face_embedding(x_hat)
        if emb is None:
            if verbose: print(f'  iter {i:03d}: no face detected')
            continue

        sim = cosine_sim(emb, target_emb)
        if sim > best_sim:
            best_sim = sim
            best_img = x_hat.cpu()

        if verbose and i % 5 == 0:
            print(f'  iter {i:03d}: sim={sim:.4f}  best={best_sim:.4f}')

        # ---- Early stopping ----
        if sim >= tau_c:
            print(f'  Early stop at iter {i}  sim={sim:.4f} >= tau_c={tau_c}')
            break

        # ---- Finite-difference gradient estimate ----
        with torch.no_grad():
            grad = torch.zeros_like(delta)
            for _ in range(n_fd):
                free_mem()
                d        = torch.randn_like(delta)
                d        = d / (d.norm() + 1e-8)
                x_plus   = generate_image((xG + delta + d).clamp(-3,3), steps=steps)
                emb_p, _ = get_face_embedding(x_plus)
                if emb_p is None: continue
                grad += (cosine_sim(emb_p, target_emb) - sim) * d
            grad /= max(n_fd, 1)

            # Gradient ascent step
            delta = delta + lr * grad

            # Project back onto L2 ball
            n = delta.norm()
            if n > epsilon:
                delta = delta * (epsilon / n)

    return best_img, best_sim


def ranked_adversary(top_n, target_emb,
                     epsilon=35.0, tau_c=0.85,
                     t_max=30, steps=15):
    """Try each candidate in rank order; stop when tau_c is met."""
    best_img, best_sim = None, -1.0
    for rank, (xG, init_sim) in enumerate(top_n):
        print(f'\n── Rank {rank+1}  init_sim={init_sim:.4f} ──')
        img, sim = attack_one(xG, target_emb,
                               epsilon=epsilon, tau_c=tau_c,
                               t_max=t_max, steps=steps)
        if img is not None and sim > best_sim:
            best_sim, best_img = sim, img
        if sim >= tau_c:
            print(f'Rank {rank+1} succeeded!')
            break
    return best_img, best_sim


print('Attack functions defined')

# ═══════════════════════════════════════════════
# CELL 10 — Demo attack on a synthetic target
# ═══════════════════════════════════════════════
free_mem()

# Generate a random target face
print('Generating target face...')
target_emb = None
while target_emb is None:
    lat        = torch.randn(1, 3, 256, 256).to(device)
    target_img = generate_image(lat, steps=25)
    target_emb, conf = get_face_embedding(target_img)

print(f'Target ready  conf={conf:.4f}')

# Step (b)
top_n = select_top_n(pool, target_emb, N=3)
free_mem()

# Step (c)
print('\nRunning attack...')
final_img, final_sim = ranked_adversary(
    top_n, target_emb,
    epsilon=35.0, tau_c=0.85,
    t_max=20, steps=15
)

# Show results
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(to_pil(target_img)); ax[0].set_title('Target',           fontsize=13); ax[0].axis('off')
ax[1].imshow(to_pil(final_img));  ax[1].set_title(f'Reconstruction\nsim={final_sim:.4f}', fontsize=13); ax[1].axis('off')
plt.suptitle('DiffMI Attack Result', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('demo_result.png', dpi=150)
plt.show()

print(f'\nFinal similarity : {final_sim:.4f}')
print(f'Decision threshold (tau_f=0.23): {"PASS" if final_sim>=0.23 else "FAIL"}')
print(f'Attack success   : {"YES" if final_sim>=0.23 else "NO"}')

# Method 2: kaggle mirror
!pip install -q kaggle
# OR use gdown from Google Drive mirror
!pip install -q gdown
import gdown
gdown.download(
    'https://drive.google.com/uc?id=1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT',
    'lfw.tgz', quiet=False
)

import os

# Check current directory
print("Files in current dir:")
print(os.listdir('.'))

# Check if tgz exists
print(f"\nlfw.tgz exists: {os.path.exists('lfw.tgz')}")
print(f"lfw folder exists: {os.path.exists('lfw')}")

import zipfile, os

print("Extracting...")
with zipfile.ZipFile('archive.zip', 'r') as z:
    z.extractall('.')

# Find where it extracted
for folder in ['lfw', 'lfw-deepfunneled', 'lfw_funneled']:
    if os.path.exists(folder):
        count = len(os.listdir(folder))
        print(f"Found: {folder}/ with {count} identities")
        lfw_dir = folder
        break

import os, zipfile

# See actual structure inside zip
with zipfile.ZipFile('archive.zip', 'r') as z:
    names = z.namelist()
    print(f"Total files in zip: {len(names)}")
    print("First 20 entries:")
    for n in names[:20]:
        print(f"  {n}")
# Check what we actually have
for root, dirs, files in os.walk('lfw-deepfunneled'):
    level = root.count(os.sep)
    if level < 3:
        print(f"{root}/ — {len(files)} files, {len(dirs)} subdirs")

# Fix nested structure
import shutil

base = 'lfw-deepfunneled'
nested = os.path.join(base, base)

if os.path.exists(nested):
    print(f"Found nested folder, fixing...")
    lfw_dir = 'lfw_fixed'
    shutil.copytree(nested, lfw_dir)
    count = len(os.listdir(lfw_dir))
    print(f"Fixed: {count} identities in {lfw_dir}/")
else:
    lfw_dir = base
    print(os.listdir(base)[:10])

t1_acc, t2_acc = evaluate_lfw(lfw_dir=lfw_dir, n_ids=10)