# Integrating MHSSA-Transformer and GCN for Enhanced Micro-expression Recognition
Pytorch Implementation of "Integrating MHSSA-Transformer and GCN for Enhanced Micro-expression Recognition"
![Overall Framework](./figure/Overall_framework.jpg)

## Abstract
Micro-expression recognition, a subfield of affective computing, plays a pivotal role in understanding genuine emotional states through subtle facial movements. Traditional methods have relied heavily on handcrafted features, which often fall short under complex conditions. Recent advancements in deep learning, particularly Graph Convolutional Networks (GCN) and Transformers, have shown promise but face challenges in balancing local and global feature learning. To address this, we introduce a novel framework that integrates the Multi-Head Spatial-aware Self-Attention (MHSSA) Transformer with GCN. This approach captures both spatial structures and temporal information in a 3D space, dynamically modeling features within and across predefined facial regions. The MHSSA-Transformer enhances fine-grained local feature extraction, while the GCN constructs adaptive adjacency matrices based on Action Unit co-occurrence and manifold distances, enabling more discriminative global feature learning. Experimental results on CASME II, SAMM, and SMIC datasets demonstrate state-of-the-art performance, with our method consistently outperforming existing approaches. This framework not only advances micro-expression recognition but also provides a robust solution for affective computing applications.

## Datasets
The datasets/ directory is structured to organize data required for running the code. If you want to reproduce the results with your own data, place your data in the relevant sub - folders as described below:
```
├─ datasets/                   # Root directory for all datasets
│  ├─ combined_datasets/       # Place onset/apex images for landmark detection here
│  ├─ Full_frame_optical_flow/ # Put full - frame optical flow data (u, v, |∇I|) here
│  └─ three_norm/              # Use this for LOSO - related data. Structure as sub_x/u_train | u_test/...
```
How to Use
For combined_datasets/: Gather your onset and apex images (used for landmark detection tasks) and place them directly in this folder. Make sure the image format (e.g., .jpg, .png) is compatible with the code that will process them.
For Full_frame_optical_flow/: If you have computed full - frame optical flow data (including the u, v components and the magnitude |∇I|), store these files in this directory. The code should be able to access and utilize these flow data for subsequent operations like feature extraction.
For three_norm/: When preparing data for the LOSO (Leave - One - Subject - Out) protocol, organize your data as sub_x/u_train | u_test/... and place it under this folder. This helps in maintaining the structure required for performing LOSO - based experiments as expected by the code.

## Installation

```
pip install -r requirements.txt   # Install dependencies
pip install dlib                  # If not automatically installed
```

>  Requires `mmod_human_face_detector.dat` and `shape_predictor_68_face_landmarks.dat` in the project root directory.

## Training

```
python train.py
```

* Performs LOSO training on each subject under `datasets/three_norm_u_v_os/`
* Saves the best model weights to `ourmodel_weights/<subject>.pth`

## Testing / Inference

```
python test.py
```

* Loads weights from `ourmodel_weights` and outputs per-subject classification reports

---

To adjust hyperparameters, edit the top section of `train.py`:

```
epochs = 100
batch_size = 128
learning_rate = 5e-5
```

## Statement
We are trying to put our paper: Integrating MHSSA-Transformer and GCN for Enhanced Micro-expression Recognition is at The Visual Computer journal. We aim to share this work to promote reproducible research in affective computing and micro-expression analysis.