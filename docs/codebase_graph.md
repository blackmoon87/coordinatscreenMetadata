# Codebase AST Dependency Graph & Architecture Map
- **Root Directory**: `/Users/blackmoon/Desktop/faceR`
- **Total Files**: 8
- **Total Classes**: 3
- **Total Functions/Methods**: 27

## File & Module Summary

### `test_incremental_visitor.py` (python)
- **Imports**: sklearn.preprocessing.LabelEncoder, pickle, time, numpy
- **Functions**: `run_incremental_visitor_simulation()`

### `prepare_dataset.py` (python)
- **Imports**: pandas, torch, facenet_pytorch.MTCNN, glob, argparse, os, re, PIL.Image
- **Functions**: `extract_label_from_filename()`, `build_dataset()`

### `core.py` (python)
- **Imports**: sklearn.ensemble.IsolationForest, numpy.typing.ArrayLike, __future__.annotations, scipy.linalg.cho_factor, scipy.linalg.cho_solve, numpy
- **Classes**: `ArrowClassifier`

### `run_pipeline.py` (python)
- **Imports**: detect, os, train, sys, time
- **Functions**: `main()`

### `train.py` (python)
- **Imports**: pandas, pickle, torch, torch.utils.data.Dataset, os, torch.utils.data.DataLoader, PIL.Image, torchvision.transforms, time, numpy
- **Classes**: `FaceDataset`
- **Functions**: `extract_features()`, `main()`

### `detect.py` (python)
- **Imports**: facenet_pytorch.InceptionResnetV1, pickle, torch, facenet_pytorch.MTCNN, glob, os, cv2, PIL.Image, torchvision.transforms, numpy
- **Classes**: `FaceRecognizerPipeline`
- **Functions**: `load_face_recognizer()`, `main()`

### `gui.py` (python)
- **Imports**: glob, os, cv2, PIL.Image, detect.FaceRecognizerPipeline, gradio, numpy
- **Functions**: `process_image_gui()`

### `stress_test_ood.py` (python)
- **Imports**: sklearn.metrics.roc_auc_score, pickle, sklearn.metrics.accuracy_score, numpy
- **Functions**: `run_stress_tests()`
