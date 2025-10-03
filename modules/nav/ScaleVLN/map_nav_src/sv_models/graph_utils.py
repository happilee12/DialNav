from collections import defaultdict
import numpy as np

MAX_DIST = 30
MAX_STEP = 10

# Global cache for vp_rel_pos_fts calculations
_vp_rel_pos_cache = {}
_vp_rel_pos_cache_miss_count = 0
_vp_rel_pos_cache_hit_count = 0

def calc_position_distance(a, b):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    return dist

def calculate_vp_rel_pos_fts(a, b, base_heading=0, base_elevation=0):
    # a, b: (x, y, z)
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # the simulator's api is weired (x-y axis is transposed)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if b[1] < a[1]:
        heading = np.pi - heading
    heading -= base_heading

    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    elevation -= base_elevation

    return heading, elevation, xyz_dist

def calculate_vp_rel_pos_fts_cached(vp1_id, vp2_id, pos1, pos2, base_heading=0, base_elevation=0):
    """
    Cached version of calculate_vp_rel_pos_fts using viewpoint IDs as keys
    
    Args:
        vp1_id: First viewpoint ID
        vp2_id: Second viewpoint ID  
        pos1: Position of first viewpoint (x, y, z)
        pos2: Position of second viewpoint (x, y, z)
        base_heading: Base heading angle
        base_elevation: Base elevation angle
    
    Returns:
        (heading, elevation, xyz_dist)
    """
    global _vp_rel_pos_cache, _vp_rel_pos_cache_miss_count, _vp_rel_pos_cache_hit_count
    
    # Create cache key from viewpoint IDs and angles
    key = (vp1_id, vp2_id, base_heading, base_elevation)
    
    if key in _vp_rel_pos_cache:
        _vp_rel_pos_cache_hit_count += 1
        # if _vp_rel_pos_cache_hit_count % 10000 == 0:
        #     print(f"VP rel pos cache HIT count: {_vp_rel_pos_cache_hit_count}, cache size: {len(_vp_rel_pos_cache)}")
        return _vp_rel_pos_cache[key]
    
    # Cache miss - calculate and store
    _vp_rel_pos_cache_miss_count += 1
    # if _vp_rel_pos_cache_miss_count % 10000 == 0:
    #     print(f"VP rel pos cache MISS count: {_vp_rel_pos_cache_miss_count}, cache size: {len(_vp_rel_pos_cache)}")
    
    result = calculate_vp_rel_pos_fts(pos1, pos2, base_heading, base_elevation)
    _vp_rel_pos_cache[key] = result
    
    return result

def calculate_vp_rel_pos_fts_cached_simple(vp1_id, vp2_id, pos1, pos2, base_heading=0, base_elevation=0):
    """
    Simplified cached version that only caches the basic calculation
    and applies heading/elevation offsets separately
    
    Args:
        vp1_id: First viewpoint ID
        vp2_id: Second viewpoint ID  
        pos1: Position of first viewpoint (x, y, z)
        pos2: Position of second viewpoint (x, y, z)
        base_heading: Base heading angle
        base_elevation: Base elevation angle
    
    Returns:
        (heading, elevation, xyz_dist)
    """
    global _vp_rel_pos_cache, _vp_rel_pos_cache_miss_count, _vp_rel_pos_cache_hit_count
    
    # Create cache key only from viewpoint IDs (without angles)
    key = (vp1_id, vp2_id)
    
    if key in _vp_rel_pos_cache:
        _vp_rel_pos_cache_hit_count += 1
        # if _vp_rel_pos_cache_hit_count % 10000 == 0:
        #     print(f"VP rel pos cache HIT count: {_vp_rel_pos_cache_hit_count}, cache size: {len(_vp_rel_pos_cache)}")
        
        # Get cached result and apply angle offsets
        cached_heading, cached_elevation, xyz_dist = _vp_rel_pos_cache[key]
        heading = cached_heading - base_heading
        elevation = cached_elevation - base_elevation
        return heading, elevation, xyz_dist
    
    # Cache miss - calculate and store
    _vp_rel_pos_cache_miss_count += 1
    # if _vp_rel_pos_cache_miss_count % 10000 == 0:
    #     print(f"VP rel pos cache MISS count: {_vp_rel_pos_cache_miss_count}, cache size: {len(_vp_rel_pos_cache)}")
    
    # Calculate without angle offsets first
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    xy_dist = max(np.sqrt(dx**2 + dy**2), 1e-8)
    xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    # Calculate base heading and elevation (without offsets)
    heading = np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
    if pos2[1] < pos1[1]:
        heading = np.pi - heading
    
    elevation = np.arcsin(dz/xyz_dist)  # [-pi/2, pi/2]
    
    # Cache the base calculation
    _vp_rel_pos_cache[key] = (heading, elevation, xyz_dist)
    
    # Apply angle offsets for return
    heading -= base_heading
    elevation -= base_elevation
    
    return heading, elevation, xyz_dist

def clear_vp_rel_pos_cache():
    """Clear the vp_rel_pos cache to free memory"""
    global _vp_rel_pos_cache, _vp_rel_pos_cache_miss_count, _vp_rel_pos_cache_hit_count
    _vp_rel_pos_cache.clear()
    _vp_rel_pos_cache_miss_count = 0
    _vp_rel_pos_cache_hit_count = 0

def get_vp_rel_pos_cache_stats():
    """Get statistics about the vp_rel_pos cache"""
    total_requests = _vp_rel_pos_cache_miss_count + _vp_rel_pos_cache_hit_count
    hit_rate = _vp_rel_pos_cache_hit_count / max(total_requests, 1) * 100
    
    return {
        'cache_size': len(_vp_rel_pos_cache),
        'cache_miss_count': _vp_rel_pos_cache_miss_count,
        'cache_hit_count': _vp_rel_pos_cache_hit_count,
        'hit_rate_percent': hit_rate,
        'memory_usage_mb': len(_vp_rel_pos_cache) * 64 / (1024 * 1024)  # Rough estimate
    }

def get_angle_fts(headings, elevations, angle_feat_size):
    ang_fts = [np.sin(headings), np.cos(headings), np.sin(elevations), np.cos(elevations)]
    ang_fts = np.vstack(ang_fts).transpose().astype(np.float32)
    num_repeats = angle_feat_size // 4
    if num_repeats > 1:
        ang_fts = np.concatenate([ang_fts] * num_repeats, 1)
    return ang_fts


class FloydGraph(object):
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


class GraphMap(object):
    def __init__(self, start_vp):
        self.start_vp = start_vp    # start viewpoint

        self.node_positions = {}             # viewpoint to position (x, y, z)
        self.graph = FloydGraph()   # shortest path graph
        self.node_embeds = {}       # {viewpoint: feature (sum feature, count)}
        self.node_stop_scores = {}  # {viewpoint: prob}
        self.node_nav_scores = {}   # {viewpoint: {t: prob}}
        self.node_step_ids = {}

    def update_graph(self, ob):
        self.node_positions[ob['viewpoint']] = ob['position']
        for cc in ob['candidate']:
            self.node_positions[cc['viewpointId']] = cc['position']
            dist = calc_position_distance(ob['position'], cc['position'])
            self.graph.add_edge(ob['viewpoint'], cc['viewpointId'], dist)
        self.graph.update(ob['viewpoint'])

    def update_node_embed(self, vp, embed, rewrite=False):
        if rewrite:
            self.node_embeds[vp] = [embed, 1]
        else:
            if vp in self.node_embeds:
                self.node_embeds[vp][0] = self.node_embeds[vp][0] + embed
                self.node_embeds[vp][1] = self.node_embeds[vp][1] + 1
            else:
                self.node_embeds[vp] = [embed, 1]
    
    def get_node_embed(self, vp):
        return self.node_embeds[vp][0] / self.node_embeds[vp][1]

    def get_pos_fts(self, cur_vp, gmap_vpids, cur_heading, cur_elevation, angle_feat_size=4):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts_cached_simple(
                    cur_vp, vp, self.node_positions[cur_vp], self.node_positions[vp],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.graph.distance(cur_vp, vp) / MAX_DIST, \
                    len(self.graph.path(cur_vp, vp)) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)

    def save_to_json(self):
        nodes = {}
        for vp, pos in self.node_positions.items():
            nodes[vp] = {
                'location': pos,    # (x, y, z)
                'visited': self.graph.visited(vp),
            }
            if nodes[vp]['visited']:
                nodes[vp]['stop_prob'] = self.node_stop_scores[vp]['stop']
                nodes[vp]['og_objid'] = self.node_stop_scores[vp]['og']
            else:
                nodes[vp]['nav_prob'] = self.node_nav_scores[vp]

        edges = []
        for k, v in self.graph._dis.items():
            for kk in v.keys():
                edges.append((k, kk))
                
        return {'nodes': nodes, 'edges': edges}
    
    
    