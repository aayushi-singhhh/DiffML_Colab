What Happened in This Implementation

The Core Idea

DiffMI is a model inversion attack — it takes a face recognition system's stored embedding (a 512-dimensional number vector representing someone's identity) and reconstructs a realistic face image from it. This proves that even systems that don't store raw photos are still a privacy risk, because the embedding contains enough identity information to recreate the person's face.
How the Attack Works

The attack uses a pretrained DDPM (a diffusion model trained on CelebA-HQ faces) as an image generator. Instead of training anything new, we treat the DDPM's input latent code as something we can optimize. We start from random Gaussian noise, run it through the DDPM to get a face, check if that face's embedding matches the target, and gradually nudge the latent code until it does. The key insight from the paper is that this is "training-free" — one fixed generator works for any target identity without retraining.
The Three Steps

Step A builds a pool of 100 "good" starting latent codes by filtering random noise through two checks — a statistical normality test ensuring the code stays Gaussian, and MTCNN face detection confirming it actually generates a real face. Step B ranks these 100 codes by how similar their generated faces already are to the target, picking the top 3 as starting points. Step C runs the actual attack — iteratively perturbing each latent code using finite-difference gradient estimation to maximize cosine similarity between the generated face and the target embedding, stopping early once similarity crosses a confidence threshold.
Why Finite Differences Instead of Backpropagation

The paper's original implementation uses APGD with direct gradients. When we tried this, the DDPM running 50 denoising steps with full gradient tracking consumed all 14GB of GPU memory and crashed. We solved this by switching to finite-difference gradient estimation — instead of computing exact gradients, we probe 8 random directions around the current latent code, measure how similarity changes in each direction, and estimate the gradient from those measurements. This uses almost no extra memory because DDPM runs under torch.no_grad() every time.
Results We Got

The attack successfully reconstructed faces with cosine similarity around 0.64, well above ArcFace's decision threshold of 0.23, meaning the reconstruction would be recognized as the same identity by the face recognition system. The generated faces visually matched the target in hair color, face shape, and general appearance. We evaluated Type-I accuracy (does reconstruction match the target image) and Type-II accuracy (does it match other images of the same person), which are the paper's primary metrics.
Why So Many Environment Errors

We spent significant time fighting package conflicts because Colab's environment has torch 2.12, numpy 2.x, and tokenizers 0.22 — all very recent versions — while diffusers and transformers have complex interdependencies that kept breaking. Every version pin we tried either conflicted with Colab's base packages or downgraded torch, which then broke GPU access. We ultimately solved it by using UNet2DModel and DDPMScheduler directly instead of DDPMPipeline, which is more version-agnostic, and by avoiding all version pinning and letting pip resolve compatible versions automatically.