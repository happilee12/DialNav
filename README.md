## TODO
- [x] Holistic inference  and checkpoints  
- [ ] Training each modules  
- [ ] Dataset  
- [ ] Dataset collection tool  



# DialNav: Holistic Inference Setup
👉 [DialNav Project Page](https://happilee12.github.io/DialNav/)

**📋 All dataset can be found in here** - All required datasets and checkpoints are available in [this Google Drive folder](https://drive.google.com/drive/folders/1MMYPP8_BiyFrxBn1kCoaLu94PErY9p6y).

This repository provides the codebase and resources for DialNav.  
Follow the steps below to prepare datasets, checkpoints, and dependencies.


## 📊 Dataset Version Management

**Dataset Update History:**
- **2025.10.03**:
  * Initial dataset upload
  * Supplementary features (connectivity, features, modules)
  * Pre-trained model checkpoints


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


### (b) RAIN Dataset
1. Download the RAIN dataset from the official link (to be provided).  
2. Place it under:
   ```
   <project_base>/dataset/RAIN/
   ```
3. For holistic inference, we specifically use:
   ```
   <project_base>/dataset/RAIN/holistic/
   ```


### (c) Checkpoints
1. Download model checkpoints from the provided link.  
2. Place them under:
   ```
   <project_base>/dataset/checkpoints/
   ```


## 2. Running Holistic Inference

1. Enter the holistic directory:
   ```bash
   cd holistic
   ```

2. Run the script:
   ```bash
   bash script/run.sh
   ```


## 3. Dependencies & Acknowledgements

### Core Codebases
This project builds upon the following open-source implementations:

- [ScaleVLN](https://github.com/wz0919/ScaleVLN.git)  
- [DUET](https://github.com/cshizhe/VLN-DUET.git)  
- [Graph-LED](https://github.com/meera1hahn/Graph_LED.git)  
- [LANA](https://github.com/wxh1996/LANA-VLN.git)  

We sincerely thank the authors of these repositories for making their code publicly available. Their contributions have been invaluable to this work.





## Simulator
We use the **Matterport3D Simulator (latest version)**.  
Follow the official instructions to install: [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator)
