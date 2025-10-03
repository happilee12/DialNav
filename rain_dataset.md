## Dataset Description

The dataset is divided into three main parts: **holistic**, **instances**, and **all**.  
Each serves a different purpose in training and evaluation.

---

##  Holistic Dataset

The **holistic dataset** is designed for **holistic evaluation** of dialog-based navigation models.  
It provides validation splits (`val_seen` and `val_unseen`) that contain complete navigation episodes.  
Each episode is represented as a JSON object with the following fields:

```json
{
    "episode_idx": 31,
    "split": "val_seen",
    "instr_id": 31,
    "start_heading": 3.141592653589793,
    "start_pano": "8b66d02c268548649cc0cc3c8a1440f9",
    "end_panos": [...],
    "scan": "Uxmj2M2itWa",
    "target": "toilet",
    "nav_steps": [...],
    "nav_idx": 0
}
```

### Field Specifications
- **`episode_idx`** *(int)*: Unique identifier for the episode.  
- **`split`** *(string)*: Dataset split (`val_seen` / `val_unseen`).  
- **`instr_id`** *(int)*: Identifier linking to the corresponding instruction/dialog.  
- **`start_pano`** *(string)*: Panorama ID where the agent starts.  
- **`end_panos`** *(list of strings)*: Goal location panoramas (may include multiple viewpoints).  
- **`scan`** *(string)*: Matterport3D scene ID.  
- **`target`** *(string)*: Object in the goal room for initial instruction  
- **`nav_steps`** *(list of strings)*: Ground-truth navigation trajectory (sequence of pano IDs).  
- **`nav_idx`** *(int)*: 0.  

---

## Segment Dataset

The **segment dataset** is designed for **dialog-grounded navigation training**.  
It contains step-wise navigation histories, dialogs, and ground-truth paths.  
Each entry represents a single dialog turn within a navigation episode.

```json
{
    "episode_idx": 0,
    "split": "train",
    "scan": "2n8kARJN3HM",
    "target": "plant",
    "end_panos": [...],
    "_start_pano_episode": "...",
    "_full_trajectory": [...],
    "_full_dialog": [...],
    "_chat_idx": 1,
    "_chat_len": 2,
    "nav_idx": 1,
    "q": "hi, im a room with a tv ...",
    "a": "Yes. Please come out of the room ...",
    "start_pano": "003f1672542542f6a4ca2903da9ac9ae",
    "nav_history": [...],
    "gt_path": [...],
    "_nav_turn": [...],
    "instr_id": "0_1"
}
```

#### Field Specifications
- **`instr_id`** *(string)*: Instruction identifier linking to dialog.  
- **`episode_idx`** *(int)*: Unique identifier for the episode.  
- **`split`** *(string)*: Dataset split (`train`, `val_seen`, `val_unseen`).  
- **`scan`** *(string)*: Matterport3D scene ID.  
- **`target`** *(string)*: Semantic goal label. 
- **`start_pano`** *(string)*: Current starting panorama for this step.  
- **`end_panos`** *(list of strings)*: Valid goal panoramas.  
- **`nav_idx`** *(int)*: Navigation step index of this segment.  
- **`q`** *(string)*: Navigator’s question at this step.  
- **`a`** *(string)*: Guide’s answer at this step.  
- **`nav_history`** *(list of strings)*: History of panoramas visited so far.  
- **`gt_path`** *(list of strings)*: Ground-truth reference path from current position to goal.  
- **`_start_pano_episode`** *(string)*: Panorama ID of the entire episode’s start location.  
- **`_full_trajectory`** *(list of strings)*: Complete trajectory of the entire episode.  
- **`_full_dialog`** *(list of QA pairs)*: Full dialog history for the entire episode.  
- **`_chat_idx`** *(int)*: Index of current dialog turn.  
- **`_chat_len`** *(int)*: Total number of dialog turns in the episode.  
- **`_nav_turn`** *(list of strings)*: Subset of navigation path relevant to this dialog turn.  
---

### All

The **all dataset** contains the **full dataset** of train and valid split of RAIN.
This is not directly used for training or inference. 
It includes metadata, trajectories, dialogs, and GUI interaction logs.  
Each entry is a complete navigation-dialog episode.
```json
{
    "meta": {
        "source": "...",
        "scan": "2n8kARJN3HM",
        "target": "plant",
        "start_pano": "...",
        "end_panos": [...],
        "nav_id": "326",
        "episode_idx": 0,
        "gui_id": "350",
        "split": "train",
        "nav_score": "4",
        "gui_score": "5",
        "total_time": 342,
        "gt_distance": 39.08,
        "path_distance": 42.62,
        "cvdn_idx": 2119,
        "path_idx": 0,
        "game_idx": 0
    },
    "gt_trajectory": [...],
    "nav_trajectory": [...],
    "stop_history": [...],
    "gui_actions": [...],
    "dialog": [...]
}
```

#### Field Specifications
- **`meta`** *(dict)*: Metadata for the episode.  
  - `source`: Original file path.  
  - `scan`: Matterport3D scene ID.  
  - `target`: Goal label.  
  - `start_pano`: Starting panorama.  
  - `end_panos`: Valid goal panoramas.  
  - `nav_id`, `gui_id`: Navigation and GUI identifiers.  
  - `episode_idx`: Episode index.  
  - `split`: Dataset split.  
  - `nav_score`, `gui_score`: Human annotation quality scores.  
  - `total_time`: Total time spent in the episode (seconds).  
  - `gt_distance`: Shortest path distance to goal.  
  - `path_distance`: Length of taken path.  
  - `cvdn_idx`: Cross-dataset alignment index.  
  - `path_idx`, `game_idx`: Additional indexing for multiple paths/games.  

- **`gt_trajectory`** *(list of strings)*: Ground-truth navigation trajectory.  
- **`nav_trajectory`** *(list of strings)*: Navigator’s executed trajectory.  
- **`stop_history`** *(list of ints)*: Indices where the agent stopped.  
- **`gui_actions`** *(list of dicts)*: GUI actions logged during data collection (navigation, room selection, trajectory exploration).  
- **`dialog`** *(list of QA pairs)*: Dialog turns (navigator question, guide answer).  
