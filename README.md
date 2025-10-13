
# DialNav: Holistic Inference Setup
This is the official repository for DialNav (ICCV '2025) [DialNav Project Page](https://happilee12.github.io/DialNav/).

This repository provides the codebase and resources for DialNav.  
Follow the steps below to prepare datasets, checkpoints, and dependencies.

<br>

## 🔗 Quick Links
- [Download RAIN Dataset](https://drive.google.com/drive/folders/1Pp_HOUDIo-uKQph18w-zqUMHcDJ6Gmg0) - Get the RAIN dataset
- [RAIN Dataset Documentation](https://github.com/happilee12/DialNav/blob/main/rain_dataset.md) - Detailed dataset explanation
- [Data Collection Tool](https://github.com/happilee12/DialNavDataCollectionTool) 

<br>

## TODO
- [x] Holistic inference  and checkpoints  
- [x] Dataset  
- [x] Dataset collection tool  
- [ ] Training each modules  

<br>

## 📊 Dataset Version Management

**Dataset Update History:**
- **2025.10.03**:
  * Initial dataset upload
  * Supplementary features (connectivity, features, modules)
  * Pre-trained model checkpoints

<br>

## 1. Dataset Preparation

### (a) Supplementary Features
1. Download the supplementary dataset:  
   [dataset.zip (Google Drive Link)](https://drive.google.com/drive/folders/1MMYPP8_BiyFrxBn1kCoaLu94PErY9p6y)

2. Unzip and place under the project root:
   ```bash
   unzip dataset.zip -d <project_base>
   ```

3. The directory structure should look like:
   ```
   <project_base>/dataset/
       ├── connectivity/
       ├── features/
       └── modules/
   ```

<br>

### (b) RAIN Dataset
1. Download the RAIN dataset:
   [RAIN (Google Drive Link)](https://drive.google.com/drive/folders/1Pp_HOUDIo-uKQph18w-zqUMHcDJ6Gmg0 )

2. Place it under:
   ```
   <project_base>/dataset/RAIN/
   ```
3. For holistic inference, we specifically use:
   ```
   <project_base>/dataset/RAIN/holistic/
   ```

<br>

### (c) Checkpoints
1. Download model checkpoints from the provided link.  
   [checkpoints (Google Drive Link)](https://drive.google.com/drive/folders/1MMYPP8_BiyFrxBn1kCoaLu94PErY9p6y)

2. Place them under:
   ```
   <project_base>/dataset/checkpoints/
   ```

<br>

## 2. Running Holistic Inference

1. Enter the holistic directory:
   ```bash
   cd holistic
   ```

2. Run the script:
   ```bash
   bash script/run.sh
   ```

<br>

## 3. Dependencies & Acknowledgements

### Core Codebases
This project builds upon the following open-source implementations:

- [ScaleVLN](https://github.com/wz0919/ScaleVLN.git)  
- [DUET](https://github.com/cshizhe/VLN-DUET.git)  
- [Graph-LED](https://github.com/meera1hahn/Graph_LED.git)  
- [LANA](https://github.com/wxh1996/LANA-VLN.git)  

We sincerely thank the authors of these repositories for making their code publicly available. Their contributions have been invaluable to this work.

<br>

## Simulator
We use the **Matterport3D Simulator (latest version)**.  
Follow the official instructions to install: [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator)
