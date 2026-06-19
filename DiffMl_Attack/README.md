# DiffMI on FaceScrub: FaceNet vs ArcFace Vulnerability Analysis

## What this is
A pilot evaluation of the DiffMI model inversion attack 
(Wang et al., https://github.com/azrealwang/DiffMI) on the FaceScrub 
dataset, a dataset not evaluated in the original DiffMI paper. 
The goal is to compare FaceNet and ArcFace face recognition models' 
vulnerability to this attack, using two evaluation frameworks:
DiffMI's own protocol (ASR/EER) and MIBench's cross-model protocol 
(Acc@1/Acc@5/feature distance).

## Setup
- Dataset: FaceScrub (10 identities × 5 images = 50 images, 
  randomly sampled with seed=0 from rajnishe/facescrub-full on Kaggle)
- Latent bank: Authors' pre-generated V=1000, τ_K=τ_D=0.999 bank
- Attack: White-box, N=3 candidate latents, eps=35, tau_C=0.98
- Platform: Kaggle Notebooks (2xT4 GPU)

## Results

### DiffMI Native Metrics
| Metric            | FaceNet | ArcFace |
|-------------------|---------|---------|
| ASR               | 0.960   | 0.960   |
| EER               | 0.051   | 0.050   |
| Type I Accuracy   | 1.000   | 1.000   |
| Type II Accuracy  | 0.950   | 0.950   |
| Avg Similarity    | 0.983   | 0.975   |
| Attack Time (s)   | 13,156  | 33,139  |
| Mean Query Cost   | ~1,073  | 1,198   |
| Margin Failures   | ~0/50   | 17/50   |

### MIBench-style Cross-Model Metrics
| Attack → Evaluator          | Acc@1 | Acc@5 | Feat. Dist |
|-----------------------------|-------|-------|------------|
| FaceNet attacked → ArcFace  | 0.74  | 0.90  | 22.61      |
| ArcFace attacked → FaceNet  | 0.96  | 0.96  | 0.83       |

## Findings
ArcFace is harder to attack efficiently: 2.5× longer attack time, 
higher query cost, and 34% margin failure rate vs ~0% for FaceNet.
FaceNet leaks more transferable identity information: reconstructions 
from ArcFace attacks are identified by FaceNet at 96% Acc@1, while 
FaceNet attack reconstructions transfer to ArcFace at only 74% Acc@1.

Both frameworks point to the same conclusion: **FaceNet is more 
vulnerable to DiffMI-style model inversion attacks than ArcFace**, 
both in terms of attack efficiency and cross-model identity leakage.

## Limitations
- Pilot scale: 10 identities / 50 images (MIBench uses 530 identities)
- FaceNet query-cost data is inferred (log lost mid-session)
- Feature distances are not cross-comparable (different embedding scales)
- FID not reported due to known instability at N=50

## References
- DiffMI: Wang et al., https://github.com/azrealwang/DiffMI
- MIBench: Qiu et al., arXiv:2410.05159, https://openreview.net/forum?id=jd2aVAqA9Q
- FaceScrub: Ng & Winkler, ICIP 2014, https://www.stefan.winklerbros.net/Publications/icip2014a.pdf