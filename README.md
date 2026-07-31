# ArrowUI — Cartesian Coordinates & Screen Metadata AI System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**ArrowUI** is an Edge-AI computer vision system that inspects digital user interfaces (Web, Mobile, Desktop) and detects 16 UI element types (`button`, `input_text`, `dropdown`, `input_checkbox`, `input_radio`, `form`, `table`, etc.), returning device-agnostic **Normalized Cartesian Coordinates `[0.0, 1.0]`** for autonomous interaction.

---

## 🌟 Key Features

- **Device-Agnostic Coords `[0.0, 1.0]`**: Normalized `(cx, cy, w, h)` coordinates work on any resolution (Mobile, Tablet, Desktop) without re-training.
- **End-to-End Fine-Tuned Vision**: EfficientNet-B0 fine-tuned on **113,169 UI element crops** from 203 URLs across 3 viewports.
- **High-Accuracy Interactive Controls**: 86.7% - 100% accuracy on interactive elements (`dropdown`: 97.8%, `button`: 92.8%, `form`: 96.3%, `textarea`: 100%).
- **100% On-Premise & Local (Edge AI)**: Light model footprint (**17 MB**) running locally with zero cloud API latency or costs.
- **Confidence-Based OOD Filtering**: No false OOD rejections on unseen websites — generalizes to any web or mobile interface.

---

## 📁 Repository Structure

```
.
├── schema.py                        # Pydantic data models & 16 UI canonical classes
├── classifier.py                    # Inference engine (ArrowUIClassifier)
├── trainer.py                       # PyTorch Fine-Tuning trainer (EfficientNet-B0)
├── extractor.py                     # Playwright DOM element extraction & coordinates
├── dataset_builder.py               # Multi-viewport dataset generator (Mobile/Tablet/Desktop)
├── click_by_model_eyes.py           # Autonomous click automation driven by model vision
├── complex_form_automation.py       # Multi-step autonomous form filling workflow
├── youtube_comment_by_model_eyes.py # YouTube comment automation
├── retest_live_public_comment.py   # Live web form retest
├── urls_production.txt              # 203 URLs used in production training
└── docs/                            # Feasibility studies, financial analysis & business models
    ├── دراسة_شاملة_مشروع_ArrowUI.md
    ├── arrow_ui_feasibility_study.md
    ├── financial_costs_and_profits.md
    └── multi_case_business_model.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/blackmoon87/coordinatscreenMetadata.git
cd coordinatscreenMetadata
pip install -r requirements.txt
playwright install chromium
```

### 2. Run Inference on Any Screenshot
```python
from classifier import ArrowUIClassifier

# Load fine-tuned ArrowUI vision model
classifier = ArrowUIClassifier(model_path="arrow_ui_classifier.pt", confidence_threshold=0.50)

# Predict elements on any screenshot
results = classifier.predict("screenshot.png")

for det in results:
    print(f"Detected {det.type} (conf: {det.confidence*100:.1f}%)")
    # Normalized Cartesian Coords [0-1]
    print(f"  Normalized: cx={det.normalized.cx}, cy={det.normalized.cy}")
    # Convert to screen pixels for 1920x1080
    px = det.normalized.to_pixels(img_width=1920, img_height=1080)
    print(f"  Pixel Coords: x1={px['x1']}, y1={px['y1']}, x2={px['x2']}, y2={px['y2']}")
```

### 3. Run Autonomous Click Automation
```bash
python3 click_by_model_eyes.py --url https://the-internet.herokuapp.com/login --target button
```

### 4. Run Multi-Step Autonomous Form Filling
```bash
python3 complex_form_automation.py
```

---

## 📊 Documentation & Studies

Detailed studies and business models available in `docs/`:
- 📄 [دراسة شاملة لمشروع ArrowUI](docs/دراسة_شاملة_مشروع_ArrowUI.md)
- 📄 [ArrowUI Feasibility Study](docs/arrow_ui_feasibility_study.md)
- 📄 [Financial Costs & Profits Analysis](docs/financial_costs_and_profits.md)
- 📄 [Multi-Case Business Model](docs/multi_case_business_model.md)

---

## 📜 License
MIT License. Free for commercial and open-source use.
