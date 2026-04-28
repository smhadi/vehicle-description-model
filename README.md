# Cross-Camera Vehicle ReID and Image-to-Text Search

**IDS 576 — Deep Learning | Final Project**  
Jacob Miller & Hadi Syed | University of Illinois Chicago

---

## Project Overview

This project builds a proof-of-concept pipeline for cross-camera vehicle re-identification (ReID) using custom data recorded at Halsted and Taylor Street in Chicago. We evaluate two complementary approaches:

1. **Appearance-based ReID (Jacob)** — FastReID with a VeRi-pretrained ResNet-50 backbone
2. **VLM Semantic Matching (Hadi)** — Qwen2-VL-7B generating structured text descriptions matched via a scoring function

**Key Results:**
| Method | Rank-1 | Rank-5 | Rank-10 |
|--------|--------|--------|---------|
| FastReID (VeRi SBS R50-ibn) | 56.52% | 86.96% | 91.30% |
| VLM Semantic Matching (Ours) | **81.1%** | **94.6%** | **97.3%** |

---

## Repository Structure

```
vehicle-description-model/
├── images/
│   ├── garage/          # Rooftop garage crops (YOLO-extracted)
│   ├── street/          # Street-level crops (YOLO-extracted)
│   ├── image_query/     # Street query images (evaluation set)
│   └── image_test/      # Garage gallery images (evaluation set)
├── data/
│   └── garage_track_summary.csv   # YOLO tracking metadata
├── notebooks/
│   ├── vehicle_inference.ipynb    # VLM batch inference on garage images
│   ├── street_matching.ipynb      # Street-to-garage matching pipeline
│   └── evaluation.ipynb           # Rank-1/5/10 evaluation
├── scripts/
│   └── batch_inference.py         # CLI batch inference script
├── results/
│   ├── garage_vlm_results.csv     # VLM descriptions for garage vehicles
│   ├── garage_to_street_matches.csv
│   └── evaluation_results.csv     # Full evaluation with rank metrics
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vehicle-description-model.git
cd vehicle-description-model
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up vLLM Server (for VLM approach)

You need a GPU machine with at least 24GB VRAM (NVIDIA L4 or A100 recommended).

```bash
# Install vLLM
pip install vllm

# Start the server
nohup python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.95 > vllm.log 2>&1 &

# Verify it's running
curl http://localhost:8000/v1/models
```

### 4. Add Your Data

Place your vehicle crop images in the appropriate directories:
- `images/image_query/` — street/query images (naming: `{id}_c{cam}_g{group}.jpg`)
- `images/image_test/` — garage/gallery images (same naming convention)
- `images/garage/` — all garage track crops (naming: `track_{id}_f{frame}_r{rep}.jpg`)
- `data/garage_track_summary.csv` — YOLO tracking summary CSV

### 5. Run Inference

**Option A: Jupyter Notebook (recommended)**
```bash
jupyter notebook notebooks/evaluation.ipynb
```
Run all cells top-to-bottom. The notebook will describe all vehicles and compute Rank-1/5/10.

**Option B: CLI script**
```bash
python scripts/batch_inference.py
```
Results saved to `results/garage_vlm_results.csv`.

---

## Data

### Sample Data
A sample of 5 matched query-gallery pairs is included in `images/image_query/` and `images/image_test/`. Filenames follow the convention `{vehicleID}_c{cameraID}_g{groupID}.jpg` — vehicles with matching first 4 digits are the same physical vehicle.

### Full Dataset
The full dataset was recorded by the project team at Halsted and Taylor Street, Chicago. To obtain the full dataset, contact the project authors. The dataset contains:
- 46 street (query) images across 37 unique vehicle identities
- 72 garage (gallery) images across the same 37 identities
- 276 garage crop images across 96 usable YOLO tracks

### Data Preprocessing Pipeline
1. Record video from both cameras simultaneously
2. Run YOLO detection and within-camera tracking
3. Filter crops by size and confidence threshold
4. Select up to 3 representative frames per track (r1, r2, r3)
5. Manually assign shared cross-camera identity labels
6. Format into VeRi-compatible folder structure for FastReID evaluation

---

## FastReID Evaluation (Jacob's Approach)

Install FastReID separately:
```bash
git clone https://github.com/JDAI-CV/fast-reid.git
cd fast-reid
pip install -r docs/requirements.txt
```

Download pretrained checkpoint:
```bash
# VeRi SBS ResNet-50 IBN (best performing)
wget https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth
```

Run evaluation:
```bash
python tools/train_net.py \
  --config-file configs/VeRi/sbs_R50-ibn.yml \
  --eval-only \
  MODEL.WEIGHTS veri_sbs_R50-ibn.pth \
  DATASETS.ROOT /path/to/your/data
```

---

## Reproducibility

- Random seeds are set in all notebooks: `random.seed(42)`, `np.random.seed(42)`
- vLLM server version: pinned in `requirements.txt`
- Model: `Qwen/Qwen2-VL-7B-Instruct` (public HuggingFace checkpoint, no fine-tuning)
- All results were generated with the exact code in this repository

---

## Requirements

See `requirements.txt` for full list. Key dependencies:
- `openai>=1.0.0` — vLLM API client
- `pandas>=2.0.0` — results processing
- `matplotlib>=3.7.0` — evaluation plots
- `jupyter>=1.0.0` — notebook execution
- `Pillow>=10.0.0` — image handling
