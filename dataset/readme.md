# Dataset Documentation

This directory contains the datasets and features required for the DialNavHolistic project. All external resources are properly attributed with download links for reproducibility.

## Visual Features

### ScaleVLN Visual Features
- **Local Path**: `/dataset/features/clip_vit-h14_mp3d_original.hdf5`
- **Description**: CLIP ViT-H/14 visual features extracted for Matterport3D scenes
- **Download**: [Hugging Face Dataset - OpenGVLab/ScaleVLN](https://huggingface.co/datasets/OpenGVLab/ScaleVLN/blob/main/features.zip)

### LANA Visual Features
- **Local Path**: `/dataset/features/CLIP-ViT-B-16-views.tsv`
- **Description**: CLIP ViT-B/16 visual features for viewpoint images in tab-separated format
- **Download**: [Google Drive](https://drive.google.com/file/d/1XPrCPLVt6mC3Mja0p2fziGSHYMjE6Z9X/view)

### CLIP Tokenizer for LANA
- **Local Path**: `/dataset/modules/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz`
- **Description**: Byte-pair encoding tokenizer vocabulary for CLIP text processing
- **Download**: [GitHub Repository - LANA-VLN](https://github.com/wxh1996/LANA-VLN/blob/main/finetune_src/clip_tokenizer/bpe_simple_vocab_16e6.txt.gz)

## GCN Localization Model

### Node Feature Generation
- **Local Path**: `/dataset/modules/node_feats`
- **Description**: Preprocessed panoramic feature representations for graph nodes
- **Preprocessing**: Refer to the original script: [Graph_LED - process-pano-feats.py](https://github.com/meera1hahn/Graph_LED/blob/main/scripts/process-pano-feats.py)

### Geodistance Nodes
- **Local Path**: `/dataset/modules/localization/geodistance_nodes`
- **Description**: Precomputed geodesic distances between navigation nodes for graph construction
- **Download**: [Google Drive](https://drive.google.com/uc?id=18RcaK3rHDKPeouEvuxMUQ4a20yW-rF2U)

---

## Acknowledgments

We extend our sincere gratitude to the following research groups and organizations for their valuable contributions to the vision-language navigation community:

- **[ScaleVLN](https://github.com/wz0919/ScaleVLN.git)** - For providing comprehensive visual features and benchmarking datasets
- **[LANA-VLN](https://github.com/wxh1996/LANA-VLN.git)** - For contributing CLIP tokenization infrastructure and visual representations  
- **[Graph_LED](https://github.com/meera1hahn/Graph_LED.git)** - For sharing graph-based localization methodologies and preprocessing tools

Their open-source contributions have been instrumental in advancing research in vision-language navigation and embodied AI. We acknowledge their commitment to open science and appreciate their efforts in making these resources accessible to the research community.