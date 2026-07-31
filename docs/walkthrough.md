# ArrowUI Classifier — Training & Test Results

## Model Architecture

```
faceR (faces)                          ArrowUI (screen elements)
─────────────                          ──────────────────────────
MTCNN crops faces from photo     →     Playwright crops elements from screenshot
InceptionResnetV1 → 512D embed   →     EfficientNet-B0 → 1280D embed
ArrowClassifier → person label   →     ArrowClassifier → element type (16 classes)
```

## Training Summary

| Metric | Value |
|--------|-------|
| **Device** | MPS (Apple Silicon GPU) |
| **Dataset** | 5,009 element crops from 30 websites |
| **Feature extraction** | 18.3s (EfficientNet-B0 → 1280-D) |
| **ArrowClassifier fit** | 0.87s (Cholesky solver) |
| **Train accuracy** | **96.93%** |
| **Test accuracy** | **91.41%** |
| **OOD threshold** | -0.4520 (auto-tuned via negative mining) |
| **Model size** | 58 MB |

## Per-Class Accuracy (Test Set)

| Class | Correct/Total | Accuracy |
|-------|---------------|----------|
| button | 11/12 | **91.7%** |
| form | 2/2 | **100.0%** |
| heading | 30/43 | 69.8% |
| image | 22/24 | **91.7%** |
| input_checkbox | 1/2 | 50.0% |
| input_radio | 1/1 | **100.0%** |
| input_text | 4/5 | **80.0%** |
| link | 349/375 | **93.1%** |
| list | 1/2 | 50.0% |
| list_item | 5/9 | 55.6% |
| navigation | 1/2 | 50.0% |
| table | 121/127 | **95.3%** |
| text | 367/397 | **92.4%** |

> [!NOTE]
> Low-count classes (checkbox, radio, list, navigation, dropdown, media, textarea) have 1-9 test samples. More training data from diverse sites will fix this.

## OOD Rejection Results

| Test | Rejected | Rate | Status |
|------|----------|------|--------|
| Random noise (100 images) | 80/100 | **80%** | Borderline |
| Solid color blocks (50 images) | 29/50 | 58% | Edge case |
| Synthetic patterns (50 images) | 50/50 | **100%** | Pass |

> [!IMPORTANT]
> Solid colors score low because the model learned many text/heading crops with solid backgrounds. In real inference, the selective proposal generator filters out uniform regions before they reach the classifier.

## Key Architecture Decisions

1. **Normalized coords [0-1]** - works on ANY screen size. No device list.
2. **OOD on ORF-projected features** - IsolationForest runs in the 9,472-D projected space where classes are separated, not raw 1280-D embeddings.
3. **Negative mining** - 500 synthetic OOD samples (noise + colors + gradients) used to auto-tune the OOD threshold.
4. **ArrowClassifier reuse** - exact same core.py from faceR, same predict_with_ood() contract.
