import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict

class Evaluator:
    def __init__(self, connectivity_dir, scans):
        self.shortest_distances = {}
        self.shortest_paths = {}
        self.connectivity_dir = connectivity_dir
        self.scans = scans
        self._load_nav_graphs()

    def load_nav_graphs(self, connectivity_dir, scans):
        def distance(pose1, pose2):
            ''' Euclidean distance between two graph poses '''
            return ((pose1['pose'][3]-pose2['pose'][3])**2\
            + (pose1['pose'][7]-pose2['pose'][7])**2\
            + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

        graphs = {}
        for scan in scans:
            with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
                G = nx.Graph()
                positions = {}
                data = json.load(f)
                for i,item in enumerate(data):
                    if item['included']:
                        for j,conn in enumerate(item['unobstructed']):
                            if conn and data[j]['included']:
                                positions[item['image_id']] = np.array([item['pose'][3],
                                        item['pose'][7], item['pose'][11]]);
                                assert data[j]['unobstructed'][i], 'Graph should be undirected'
                                G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
                nx.set_node_attributes(G, values=positions, name='position')
                graphs[scan] = G
        return graphs
    
    def _load_nav_graphs(self):
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = self.load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
    
    def _get_nearest(self, shortest_distances, goal_id, viewpoint_ids):
        near_id = viewpoint_ids[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in viewpoint_ids:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id


    def _cal_dtw(self, shortest_distances, prediction, reference, success=None, threshold=3.0):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = shortest_distances[prediction[i-1]][reference[j-1]]
                dtw_matrix[i][j] = cost + best_previous_cost

        dtw = dtw_matrix[len(prediction)][len(reference)]
        ndtw = np.exp(-dtw/(threshold * len(reference)))
        if success is None:
            success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
        sdtw = success * ndtw

        return {
            'DTW': dtw,
            'nDTW': ndtw,
            'SDTW': sdtw
        }

    def _cal_cls(self, shortest_distances, prediction, reference, threshold=3.0):
        def length(nodes):
            return np.sum([
                shortest_distances[a][b]
                for a, b in zip(nodes[:-1], nodes[1:])
            ])

        coverage = np.mean([
            np.exp(-np.min([  # pylint: disable=g-complex-comprehension
                shortest_distances[u][v] for v in prediction
            ]) / threshold) for u in reference
        ])
        expected = coverage * length(reference)
        score = expected / (expected + np.abs(expected - length(prediction)))
        return coverage * score
    
    def get_shortest(self, scan, start_pano, end_panos):
        shortest_distances = self.shortest_distances[scan]
        # shortest_paths = self.shortest_paths[scan]
        shortest_distance = np.min([shortest_distances[start_pano][end_pano] for end_pano in end_panos])
        return shortest_distance
        

    def _eval_item(self, scan, predicted_path, end_panos):
        scores = {}
        shortest_distances = self.shortest_distances[scan]
        shortest_paths = self.shortest_paths[scan]

        flattend_path = sum(predicted_path, [])
        start_pano = flattend_path[0]
        shortest_goal = self._get_nearest(shortest_distances, start_pano, end_panos)
        shortest_path = shortest_paths[start_pano][shortest_goal]
        shortest_distance = shortest_distances[start_pano][shortest_goal]
        trajectory_distance = np.sum([shortest_distances[a][b] for a, b in zip(flattend_path[:-1], flattend_path[1:])])

        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(flattend_path[:-1], flattend_path[1:])])
        scores['trajectory_steps'] = len(flattend_path) - 1
        scores['success'] = float(flattend_path[-1] in end_panos)
        scores['oracle_success'] = float(any(x in end_panos for x in flattend_path))
        scores['spl'] = scores['success'] * shortest_distance / max(trajectory_distance, shortest_distance, 0.01)
        scores['nav_error'] = np.min([shortest_distances[flattend_path[-1]][end_pano] for end_pano in end_panos])
        scores['gp'] = shortest_distance - scores['nav_error']
        ERROR_MARGIN = 3
        scores.update(
            self._cal_dtw(shortest_distances, flattend_path, shortest_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = self._cal_cls(shortest_distances, flattend_path, shortest_path, ERROR_MARGIN)
        scores['gt_lengths'] = int(shortest_distance)

        return scores
    
    def eval_metrics(self, navigation_output):
        metrics = defaultdict(list)

        for item in navigation_output:
            scan = item['scan']
            pred_path = item['path']
            end_panos = item['end_panos']
            dialogs = [item for item in item['navigation_detail'] if item['ask']]
            dtc = len(dialogs)
            les = [item['loc_error'] for item in dialogs]
            
            scores = self._eval_item(scan, pred_path, end_panos)
            for k, v in scores.items():
                metrics[k].append(v)
            metrics['dtc'].append(dtc)
            metrics['le'].extend(les)
            
        avg_metrics = {
            'count': len(navigation_output),
            'gt_lengths': np.mean(metrics['gt_lengths']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'gp': np.mean(metrics['gp']),
            'nav_error': np.nanmean(metrics['nav_error']),
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
            'le': np.mean(metrics['le']),
            'dtc': np.mean(metrics['dtc'])
        }
        
        return avg_metrics, metrics
    