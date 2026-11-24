# Speaker Notes: Text-to-Image Generation Presentation
**Total Time: 20-25 minutes | 3 Presenters (7-8 min each)**

---

## PERSON A: Data & Architecture (~7 minutes)

### Slide: Motivation (1 min)
- **Opening**: "Today we'll walk you through our work on text-to-image generation using diffusion models."
- Explain the problem: generating domain-specific images requires fine-tuning
- Preview: We'll cover data, architecture, and training

### Slide: Oxford-102 Flowers Dataset (1.5 min)
- "We're using the Oxford-102 Flowers benchmark dataset with 8,189 images"
- Unusual split: 75% in test set (this is intentional for benchmarking)
- Key point: Only 1,020 training images — small dataset, need careful adaptation

### Slide: Data Processing Pipeline (1 min)
- Walk through each step: raw → resize → normalize → caption → split
- "BLIP generates natural language descriptions for each image"
- Result: Consistent, augmented dataset ready for training

### Slide: SD v1.5 Architecture (2 min)
- "Stable Diffusion has 3 main components..."
- Diagram explanation:
  - CLIP: 77 tokens, frozen, semantic understanding from text
  - VAE: Compresses 512×512 to 64×64×4 (48× compression)
  - U-Net: 860M parameters, denoises guided by text
- "The latent space compression is key — allows training on consumer GPUs"

### Slide: Diffusion Process (1.5 min)
- **Forward**: "During training, we add noise step-by-step, then predict it back"
- Show equation: z_t combines original + noise weighted by schedule
- **Reverse**: "During inference, we start from pure noise and iteratively remove it"
- "This reverse process is guided by the text prompt to generate relevant images"

### Transition**: "Person B will now explain how we efficiently adapt this huge model..."

---

## PERSON B: Training & LoRA (~7 minutes)

### Slide: Why LoRA? (1.5 min)
- "Fine-tuning 860M parameters is too expensive"
- Introduce LoRA: inject low-rank matrices into attention layers
- Table: Shows 50-100× parameter reduction
- "Only 1.6M trainable params for rank 16 — just 0.19% of the model"

### Slide: Training Pipeline (1.5 min)
- Walk through 7-step process:
  1. Sample random timestep t
  2. VAE encode the image
  3. Add noise
  4. CLIP encode the caption
  5. U-Net (with LoRA) predicts the noise
  6. MSE loss
  7. Update only LoRA weights
- "Mixed precision: U-Net stays fp32 for stability, VAE/CLIP in fp16"
- Result: 50% memory savings

### Slide: Training Results (1.5 min)
- "We trained 4 different LoRA models with ranks 4, 8, 16, 32"
- **Major finding**: Rank 32 achieves FID 75.26 — significantly better than rank 16 (85.49)
- "This was surprising! More capacity is needed for this domain."
- Show loss convergence: sharp drop then stable

### Slide: Loss Curve (1 min)
- "Training is stable across 2000 steps"
- Checkpoints saved at 500, 1000, 1500, 2000
- "Later, we'll see that step 500 is actually optimal"

### Transition**: "Person C will now show our comprehensive evaluation and which settings work best..."

---

## PERSON C: Evaluation & Ablations (~7 minutes)

### Slide: Evaluation Metrics (1 min)
- Explain 3 metrics:
  - FID: "Quality and diversity vs real flowers" (lower better)
  - KID: "More reliable on small datasets like ours"
  - CLIP Score: "How well image matches the prompt" (higher better)
- "We evaluate on 200 prompts × 5 seeds = 1,000 images for fair comparison"

### Slide: A01 Baseline vs LoRA (0.5 min)
- "Does fine-tuning actually help?"
- FID: 87.25 → 85.49 (2% improvement, modest)
- KID: 0.0357 → 0.0310 (13% improvement, strong)
- "KID shows better distribution alignment than FID"

### Slide: A02 Guidance Scale (1 min)
- "Guidance scale controls how much the model follows the prompt"
- Show curve: U-shaped, optimal at 5.0 (FID 78.96)
- Too low (1.0): poor quality | Too high (15.0): mode collapse
- "Default is 7.5, but we found 5.0 is optimal for flowers"

### Slide: A03-A04 Steps & Scheduler (1 min)
- Steps: 10 fast (4.5s), 20 balanced (8.6s best FID), 30 degrading (12.9s)
- Schedulers: DDIM best (80.46), Euler competitive (82.26), PNDM weak
- "Scheduler choice matters significantly"

### Slide: A05 LoRA Rank ⭐ (1.5 min)
- **This is the standout finding**
- "Rank 32 achieves FID 75.26 — the best result across ALL our experiments"
- Rank 4: 82.03, Rank 8: 79.16, Rank 16: 85.49, **Rank 32: 75.26**
- "Non-monotonic relationship suggests this domain needs more capacity"
- "If we had to pick one result, this is it"

### Slide: A06 Training Steps (0.5 min)
- "Should we train longer?"
- Step 500: FID 79.89 (best), Step 1000: 82.16, Step 2000: 85.49
- "Training longer actually hurts FID — we're overfitting"
- "Solution: Use early checkpoint at step 500"

### Slide: A07-A08 Negative Prompts & SDXL (0.5 min)
- Negative prompts: Hurt FID (85.49 → 86.06)
  - Generic negatives reduce diversity for specialized domains
- SDXL: Achieves same FID as SD v1.5 baseline (87.25)
  - Our LoRA-tuned SD v1.5 beats SDXL zero-shot ✓

### Slide: Summary (0.5 min)
- Best settings table: Rank 32, Guidance 5.0, DDIM, 20-30 steps, Step 500 checkpoint
- **FID 75.26 is achievable**

### Slide: Contributions & Conclusions (1 min)
- Summarize all 3 parts
- Key takeaway: "Domain matters. Specialized fine-tuning outperforms general models."
- "Rank 32 LoRA needed here — efficiency is secondary to quality"
- "Early stopping (step 500) prevents overfitting"

### Final Slide: Thank You
- "Questions?"
- Mention dataset, model, method used

---

## Timing Breakdown

| Person | Slides | Time |
|--------|--------|------|
| A | 1-7 (Motivation through Diffusion) | 7 min |
| B | 8-12 (Why LoRA through Loss Curve) | 7 min |
| C | 13-23 (Metrics through Conclusions) | 7 min |
| Q&A | - | 2-3 min |
| **Total** | **23 slides** | **23 min** |

---

## Presentation Tips

### General
- **Pacing**: 1-2 min per slide maximum
- **Interaction**: Ask "Any questions so far?" between sections
- **Eye contact**: Make eye contact with audience, not just slides
- **Transitions**: Use "Person B/C, take it away" for hand-offs

### For Diagrams
- When showing architecture diagram: Point to each component, explain once
- When showing tables: Call out the key number (e.g., "Look at rank 32: FID 75.26")
- When showing loss curve: "Notice how it drops sharply then plateaus"

### For Data & Numbers
- Always say "lower is better" for FID (doesn't become obvious)
- Always say "higher is better" for CLIP Score
- Use percentages to make improvements sound significant ("13% improvement in KID")

### Handling Questions
- If asked "Why is rank 32 better?" → "We're not entirely sure, but the data shows it needs more capacity. This domain has complex textures and details."
- If asked "Will this work on other datasets?" → "Good question! This was optimized for flowers. Each domain probably has different optimal rank."

---

## What NOT to Do
- ❌ Read slides verbatim
- ❌ Talk too fast (slow down!)
- ❌ Spend more than 2 min on any single slide
- ❌ Say "um" or "uh" (take a breath instead)
- ❌ Turn your back to the audience

---

## Backup Slides (Optional)
If audience asks for more detail:
- Latent space visualization (what does VAE compress?)
- Different guidance scales visual comparison
- Full ablation table
- Failure cases (blurry, wrong species, etc.)

---

**Good luck! You've done great work. Keep energy high, explain clearly, let the data speak for itself.** 🎓
