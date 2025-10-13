import os
import json
import jsonlines
import h5py
import networkx as nx
import math
import numpy as np


class ImageFeaturesDB(object):
    def __init__(self, img_ft_file, image_feat_size, use_clip16=False):
        self.image_feat_size = image_feat_size
        self.img_ft_file = img_ft_file
        self._feature_store = {}
        self.clip16_fts = {}
        self.use_clip16 = use_clip16
        if self.use_clip16:
            tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
            import csv, base64    
            with open(self.img_ft_file, "r") as tsv_in_file:     # Open the tsv file.
                reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
                for item in reader:
                    long_id = item['scanId'] + "_" + item['viewpointId']
                    self.clip16_fts[long_id] = np.frombuffer(base64.decodestring(item['features'].encode('ascii')),
                                                    dtype=np.float32).reshape((36, -1))   # Feature of long_id is (36, 2048)
            print(f'loaded feature from {self.img_ft_file}')

    def get_image_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.use_clip16:
            ft = self.clip16_fts[key]
            return ft
        if key in self._feature_store:
            ft = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._feature_store[key] = ft
        return ft


def load_instr_datasets(anno_path_list):
    data = []
    for anno_path in anno_path_list:
        if anno_path.endswith('json'):
            with open(os.path.join(anno_path)) as f:
                new_data = json.load(f)
        elif anno_path.endswith('jsonl'):
            with jsonlines.open(anno_path) as f:
                new_data = [item for item in f]
        else:
            raise NotImplementedError('unsupported annotation file %s' % anno_path)

        # Join
        print(f"Loaded {len(new_data)} data from {anno_path}")
        data += new_data
    return data


def construct_instrs(anno_paths, tokenizer=None, max_given_len=0, max_instr_len=512, max_action_len=50, use_clip16=False,  caption_type='question', target_path='gt_path', debug=False):
    print("construct_instrs", anno_paths, "caption_type", caption_type, "target_path", target_path)

    
    data = []
    anno_path_list = anno_paths.split(',')
    datasets = load_instr_datasets(anno_path_list)


    ### check if to replace target_path with _nav_turn
    if target_path == 'instruction_path' \
        and caption_type == 'answer' \
        and 'instruction_path' not in datasets[0]:
        if '_nav_turn' in datasets[0]:
            target_path = '_nav_turn'
            print(f"No target path {target_path} in {anno_paths}, using _nav_turn instead")
        else:
            raise ValueError(f"No target path {target_path} in {anno_paths}")

    for i, item in enumerate(datasets):
        new_item = dict(item)
        new_item['path_id'] = new_item['instr_id'] 
        new_item['heading'] = 3.141592653589793
        if caption_type == 'question' and 'q' not in new_item:
            continue
        if caption_type == 'answer' and 'a' not in new_item:
            continue
        
        # GIVEN = False
        # if max_given_len > 0:
        #     GIVEN = True
        # if GIVEN:
        #     if caption_type == 'question':
        #         new_item['given'] = "Target : " + new_item['target'] + " Question : " # given target
        #     elif caption_type == 'answer':
        #         new_item['given'] = "Question : " + new_item['q'] + " Answer : " # given target


        if caption_type == 'question': # generate question
            new_item['instruction'] = new_item['q'] 
            # nav_idx = new_item['nav_idx']
            # nav_history = new_item['nav_history']
            new_item['path'] = new_item[target_path][-max_action_len:]
            new_item['path_to_goal'] = False
            # print("new_item['path']", new_item['path'])
        elif caption_type == 'answer': # generate question
            new_item['instruction'] = new_item['a'] 
            new_item['path'] = new_item[target_path][:max_action_len]
            new_item['path_to_goal'] = False
            if target_path == '_nav_turn':
                if item['_chat_idx'] == new_item['_chat_len']:
                    new_item['path_to_goal'] = True
            else:
                if len(new_item[target_path]) < max_action_len:
                    new_item['path_to_goal'] = True
                
        
        if use_clip16:
            clip_pad_token_id = 0
            # if GIVEN:
            #     encoded = tokenizer.encode(new_item['given'])[:max_given_len]
            #     padding_length = max_given_len - len(encoded)
            #     new_item['given_encoding'] = [clip_pad_token_id] * padding_length + encoded
            new_item['instr_encoding'] = tokenizer.encode(new_item['instruction'])[:max_instr_len] # NOTE

        keys_to_delete = [
            'answer', 
            'previous_dialog', 
            'nav_history', 
            'question_enc', 
            'question', 
            'answer_enc', 
            'target_encoding', 
            'nav_steps'
        ]
        for key in keys_to_delete:
            if key in new_item:
                del new_item[key]


        data.append(new_item)
        if debug and len(data) > 10:
            break

    print("loaded data from ", anno_paths)
    print("dataset size", len(datasets))
    print("data size", len(data))
    return data


def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2 \
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2 \
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def angle_feature(heading, elevation, angle_feat_size):
    return np.array(
        [math.sin(heading), math.cos(heading), math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
        dtype=np.float32)


def new_simulator(connectivity_dir, scan_data_dir=None):
    import MatterSim

    # Simulator image parameters
    WIDTH = 640
    HEIGHT = 480
    VFOV = 60

    sim = MatterSim.Simulator()
    if scan_data_dir:
        sim.setDatasetPath(scan_data_dir)
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.initialize()

    return sim


def get_point_angle_feature(sim, angle_feat_size, baseViewId=0, minus_elevation=False):
    feature = np.empty((36, angle_feat_size), np.float32)
    base_heading = (baseViewId % 12) * math.radians(30)
    if minus_elevation:
        base_elevation = (baseViewId // 12 - 1) * math.radians(30)
    else:
        base_elevation = 0

    for ix in range(36):
        if ix == 0:
            sim.newEpisode(['ZMojNkEp431'], ['2f4d90acd4024c269fb0efe49a8ac540'], [0], [math.radians(-30)])
        elif ix % 12 == 0:
            sim.makeAction([0], [1.0], [1.0])
        else:
            sim.makeAction([0], [1.0], [0])

        state = sim.getState()[0]
        assert state.viewIndex == ix

        heading = state.heading - base_heading
        elevation = state.elevation - base_elevation

        feature[ix, :] = angle_feature(heading, elevation, angle_feat_size)
    return feature


def get_all_point_angle_feature(sim, angle_feat_size, minus_elevation=False):
    return [get_point_angle_feature(
        sim, angle_feat_size, baseViewId, minus_elevation=minus_elevation
    ) for baseViewId in range(36)]
