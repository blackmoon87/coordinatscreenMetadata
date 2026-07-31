# faceR — ArrowAI Face Detection & Recognition Pipeline

A fast face detection and high-accuracy identity recognition system utilizing **InceptionResnetV1** for feature embeddings, **MTCNN** for face detection, and a custom **ArrowClassifier** (Orthogonal Random Features) for classification.

---

## ⚙️ How It Works

The **faceR** architecture operates in three main stages:

```mermaid
graph LR
    A["Raw / Full Images"] --> B["1. MTCNN Face Detector"]
    B --> C["2. InceptionResnetV1 (512D Embedding)"]
    C --> D["3. ArrowClassifier (8192D ORF)"]
    D --> E["Predicted Identity & Bounding Box"]
```

1. **Face Detection & Alignment (`MTCNN`)**
   - Scans input photos to detect face bounding boxes and facial landmarks.
   - Aligns and crops face regions, normalizing them to $160 \times 160$ resolution.

2. **Facial Feature Embedding (`InceptionResnetV1`)**
   - Passes cropped face tensors through a pre-trained **VGGFace2** deep ResNet model.
   - Extracts a dense $512$-dimensional embedding vector representing biometric facial features.

3. **High-Speed Classification (`ArrowClassifier`)**
   - Expands $512$-D embeddings into an $8192$-dimensional feature space using **Orthogonal Random Features (ORF)** and skip connections.
   - Trains via closed-form regularized least-squares (Cholesky solver) for ultra-fast fitting and high-accuracy identity predictions.

---

## 📁 Project Structure & File Explanations

```text
faceR/
├── core.py                      # Custom ArrowClassifier algorithm implementation
├── prepare_dataset.py           # Auto-extract faces from raw images & build Dataset.csv
├── train.py                     # Dataset loading, feature extraction & model training script
├── detect.py                    # End-to-end MTCNN face detection & recognition pipeline
├── gui.py                       # Interactive Gradio web interface for face recognition
├── run_pipeline.py              # Master runner script (trains model + runs detection test)
├── requirements.txt             # Python dependencies package list
├── arrow_face_classifier.pkl    # Serialized trained model & label encoder weights
├── Dataset.csv                  # CSV mapping face image filenames to identity labels
├── Faces/                       # Directory containing cropped facial image dataset
├── Original Images/             # Directory containing full-resolution test images by identity
└── output/                      # Directory storing annotated detection output images
```

### 📄 File & Directory Descriptions

- **`core.py`**
  Contains the implementation of `ArrowClassifier`. It uses Orthogonal Random Features (ORF) and skip connections to project face embeddings into a higher-dimensional space and solves for classification weights using regularized least-squares (Cholesky factorization).

- **`train.py`**
  Extracts 512-dimensional facial embeddings from `Faces/` using pretrained `InceptionResnetV1` (VGGFace2), splits the dataset, trains the `ArrowClassifier`, and saves the model to `arrow_face_classifier.pkl`.

- **`detect.py`**
  Implements `FaceRecognizerPipeline`. Uses `MTCNN` to locate faces in images, crops & passes them to `InceptionResnetV1` to extract embeddings, predicts identities with `ArrowClassifier`, and draws bounding boxes with confidence scores. Saves annotated images to `output/`.

- **`gui.py`**
  Launches a web GUI using **Gradio**. Allows users to upload images or select samples, adjust detection confidence thresholds, and view real-time bounding box annotations and recognition reports.

- **`run_pipeline.py`**
  Automated end-to-end execution script. Sequentially triggers `train.py` to train/update the classifier and `detect.py` to evaluate performance on test images.

- **`arrow_face_classifier.pkl`**
  Binary pickle file storing the trained `ArrowClassifier` model instance and `LabelEncoder` mapping.

- **`Dataset.csv`**
  Metadata file pairing cropped face image IDs with their corresponding person identity labels.

- **`Faces/`**
  Folder containing cropped face images used for training the feature extraction and classification pipeline.

- **`Original Images/`**
  Folder containing full uncropped test photos organized by identity folders for detection testing.

- **`output/`**
  Folder where result images with detected face bounding boxes and identity banners are saved.

---

## 🏷️ Dataset File Naming Conventions

### 1. Cropped Face Images (`Faces/Faces/`)
Images used for feature extraction and training are named following the pattern:
```text
{Person_Name}_{Index}.jpg
```
- **`{Person_Name}`**: The identity/name of the individual (e.g., `Robert Downey Jr`, `Billie Eilish`).
- **`_`**: Underscore delimiter separating the identity name and the sample index.
- **`{Index}`**: A unique integer starting from `0` representing different cropped variations of that person's face.
- **Example**: `Robert Downey Jr_87.jpg`, `Akshay Kumar_0.jpg`, `Billie Eilish_3.jpg`

### 2. CSV Mapping (`Dataset.csv`)
Pairs each cropped image filename with its target class label:
| Column | Description | Example |
| :--- | :--- | :--- |
| `id` | Exact image filename inside `Faces/Faces/` | `Robert Downey Jr_87.jpg` |
| `label` | Target identity string used for classification | `Robert Downey Jr` |

### 3. Full Original Test Images (`Original Images/Original Images/`)
Organized by subfolders per identity containing original full-resolution photos:
```text
Original Images/Original Images/{Person_Name}/{Person_Name}_{Index}.jpg
```
- **Folder Name**: Name of the identity (e.g., `Original Images/Original Images/Brad Pitt/`).
- **Image Filename**: Follows `{Person_Name}_{Index}.jpg` (or `.png`).

### 4. Detection Output Files (`output/`)
Saved annotated output photos generated by `detect.py`:
```text
detected_{Person_Name_With_Underscores}.jpg
```
- **Example**: `detected_Akshay_Kumar.jpg`, `detected_Robert_Downey_Jr.jpg`

---


---

## 🛠️ Prerequisites & Environment Setup

### 1. Requirements
- **Python**: `Python 3.8` or higher
- **GPU (Optional)**: NVIDIA GPU with CUDA support for accelerated face detection & feature extraction (CPU is supported out-of-the-box).

### 2. Installation
Open your terminal and install all required Python dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Usage Guide (Step-by-Step)

### Step 1: Prepare Your Dataset (If starting from scratch)
If you only have raw photos in a folder named `imgs/`:
```bash
python3 prepare_dataset.py --input_dir imgs --identities person1Name person2Name
```
> **What this does**: Automatically detects faces, crops them to `Faces/Faces/`, and creates `Dataset.csv`.

### Step 2: Train the Classifier Model
Train the `ArrowClassifier` on your cropped face dataset:
```bash
python3 train.py
```
> **Output**: Saves the trained model weights to `arrow_face_classifier.pkl`.

### Step 3: Run Identity Recognition on Images
Test face detection and identity predictions on images:
```bash
python3 detect.py
```
> **Output**: Saves annotated output photos with green bounding boxes to the `output/` directory.

### Step 4: Launch Interactive Web Interface
Open a user-friendly browser interface for live testing:
```bash
python3 gui.py
```
> **Access GUI**: Opens automatically in your browser at `http://127.0.0.1:7860`.

---

## ❓ FAQ & Troubleshooting

> [!NOTE]
> **Q: "ModuleNotFoundError: No module named 'facenet_pytorch'" or similar error?**  
> **A:** Make sure you installed dependencies by running `pip install -r requirements.txt`.

> [!TIP]
> **Q: Where are the output photos saved after running `detect.py`?**  
> **A:** Look inside the `output/` folder created in your project directory.

> [!IMPORTANT]
> **Q: What if no face is detected in an image?**  
> **A:** Ensure the photo has clear lighting and the person's face is unobstructed. In `gui.py`, you can also lower the *Detection Confidence Threshold* slider.

