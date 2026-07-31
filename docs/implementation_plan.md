# ArrowUI Production Upgrade Plan

## Problem

91.4% accuracy on 5,009 crops from 30 sites is **not production-grade**. Root causes:

| Issue | Current | Target |
|-------|---------|--------|
| Training data | 5,009 crops, 30 sites | **50K+ crops, 200+ sites** |
| Class balance | text=1986, dropdown=1 | **min 200 per class** |
| Backbone | Frozen EfficientNet (dumb feature extractor) | **Fine-tuned last 3 blocks** |
| Augmentation | None | **Color/brightness/scale jitter** |
| Viewports | Desktop only (1920×1080) | **Mobile + Tablet + Desktop** |
| Accuracy | 91.4% | **98%+** |

## Proposed Changes

### 1. Massive Dataset Scaling

#### [MODIFY] [dataset_builder.py](file:///Users/blackmoon/Desktop/faceR/CartesianCoordinatesScreenMetadata/dataset_builder.py)

- Add multi-viewport scraping: each URL scraped at 3 viewports:
  - `375×667` (iPhone SE — mobile)
  - `768×1024` (iPad — tablet) 
  - `1920×1080` (Desktop)
- This 3x the data AND teaches the model that the same button looks different at different sizes

#### [NEW] urls_production.txt

200+ URLs targeting balanced class coverage:
- **Forms-heavy** sites (login pages, signup forms, contact forms) → input_text, checkbox, radio, dropdown, textarea, button
- **Media-heavy** sites (YouTube, Vimeo, Spotify) → media, image
- **Navigation-heavy** sites (news sites, docs sites) → navigation, list, list_item
- **Table-heavy** sites (data tables, pricing pages) → table
- **General** (Wikipedia, GitHub, StackOverflow, HN) → text, link, heading

---

### 2. Data Augmentation

#### [MODIFY] [trainer.py](file:///Users/blackmoon/Desktop/faceR/CartesianCoordinatesScreenMetadata/trainer.py)

Add augmentation transforms during training (NOT during feature extraction):

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Plus class-weighted oversampling for rare classes using `WeightedRandomSampler`.

---

### 3. Fine-Tuned EfficientNet-B0 (replaces frozen + ArrowClassifier)

#### [MODIFY] [trainer.py](file:///Users/blackmoon/Desktop/faceR/CartesianCoordinatesScreenMetadata/trainer.py)

**Replace** frozen-features + ArrowClassifier with end-to-end fine-tuning:

```
EfficientNet-B0 (blocks 1-6 frozen, blocks 7-8 + classifier unfrozen)
    ↓
Linear(1280 → 256) + ReLU + Dropout(0.3)
    ↓
Linear(256 → 16)  # 16 UI classes
    ↓
CrossEntropyLoss (class-weighted)
```

- **Why**: ArrowClassifier is a single linear layer on frozen features. Fine-tuning lets the model actually LEARN what makes a "button" vs "link" vs "text" in visual space, not just in pre-trained ImageNet feature space.
- **Epochs**: 15-20 with early stopping on validation loss
- **LR**: 1e-4 for unfrozen blocks, 1e-3 for classifier head
- **Scheduler**: CosineAnnealingLR

> [!IMPORTANT]
> This replaces ArrowClassifier for the UI classification task. ArrowClassifier is great for face recognition (few-shot, closed-set). For UI elements across infinite websites, we need a model that has learned to distinguish UI visual patterns, not just linear separation on frozen ImageNet features.

#### [MODIFY] [classifier.py](file:///Users/blackmoon/Desktop/faceR/CartesianCoordinatesScreenMetadata/classifier.py)

Update inference to use the fine-tuned model directly:
- Load fine-tuned EfficientNet-B0 + classifier head
- Forward pass → softmax → class prediction + confidence
- No ArrowClassifier, no IsolationForest

---

### 4. ArrowClassifier Still Available

> [!NOTE]
> ArrowClassifier stays in `core.py` — it's still the right choice for faceR (face recognition). For UI classification on infinite websites, the fine-tuned approach is better because:
> - Face recognition = closed set (known people). ArrowClassifier excels here.
> - UI classification = open set (infinite websites). Need learned features, not memorized features.

## Verification Plan

### Automated Tests
```bash
# 1. Build production dataset (200+ URLs × 3 viewports)
python3 dataset_builder.py --batch urls_production.txt --multi-viewport

# 2. Train fine-tuned model
python3 trainer.py --dataset ./dataset --epochs 20 --fine-tune

# 3. Test accuracy + per-class breakdown
python3 test_ood.py

# 4. Cross-site generalization test (hold out 20% of SITES, not samples)
python3 test_generalization.py
```

### Manual Verification
- Run inference on screenshots from 10 completely new websites
- Verify correct classification + bounding boxes visually
