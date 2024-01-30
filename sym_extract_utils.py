import random
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler



def umb(sig, thresh):
    if sig['kMin']!=0:
        return abs(sig['kMax']/sig['kMin']) < thresh
    else:
        pass
    
def prune_points(signatures, thresh):
    pruned = [s for s in signatures if umb(s, thresh)]
    return pruned


def random_sample(v, num_samples):
    indices = list(range(len(v)))
    random.shuffle(indices)
    sampled = [v[idx] for idx in indices[:num_samples]]
    return sampled

def flatten_all(signatures):
    all_flattened = []

    for sig in signatures:
        all_flattened.append([sig['kMin'], sig['kMax']])

    return all_flattened


def transform(a,b):
    origin_index = a['index']
    image_index = b['index']
    s = (a['kMin'] / b['kMin']+ a['kMax'] / b['kMax']) / 2
    s = 1
        
    A = np.vstack((a['minCurv'], a['maxCurv'], a['normal']))
    B = np.vstack((b['minCurv'], b['maxCurv'], b['normal']))

    R = np.dot(B, A.T)
    t = b['ptcoor'] - s * np.dot(R, a['ptcoor'])

    return {'t': t, "s": s, "R": R, "orig": origin_index, "image": image_index}

def transform_point(point):
    if len(point) != 9:
        raise ValueError("Invalid point size")

    s = point[0]
    angles = point[1:4]
    rotation_matrix = Rotation.from_euler('xyz', angles, degrees=False).as_matrix()
    R = rotation_matrix
    t = np.array(point[4:7]).reshape((3, 1))
    origin_index = int(point[7])
    image_index = int(point[8])
    
    return {'t': t, "s": s, "R": R, "orig": origin_index, "image": image_index}

def transform_to_point(trans):
    t_vector = list(trans["t"].flatten())
    ea = Rotation.from_matrix(trans['R']).as_euler('xyz')
    ea_vector = list(ea)
    return [trans["s"], ea_vector[0], ea_vector[1], ea_vector[2],
        t_vector[0], t_vector[1], t_vector[2],
        float(trans["orig"]), float(trans["image"])]


def build_pairing_kd_tree(signatures, diag, rigid=True, filename="final_transf_space.txt"):
    transf = []
    with open(filename, 'w') as fs:
        num_samples = min(len(signatures), 100)
        samples = random_sample(signatures, num_samples)
        datapoints = np.array(flatten_all(signatures))
        query = np.array(flatten_all(samples))
        
        radius = diag
        num_neighbors = 128
        nbrs = NearestNeighbors(radius=radius, n_neighbors=num_neighbors, algorithm='kd_tree').fit(datapoints)
        indices = nbrs.radius_neighbors(query, return_distance=False)

        for i in range(len(samples)):
            neighbors = indices[i]
            p_a = samples[i]
            for j in neighbors:
                p_b = signatures[j]
                if p_a['index'] == p_b['index']:
                    continue
                t = transform(p_a, p_b)
                fs.write(str(t) + '\n')
                transf.append(t)
    return transf



def run_clustering(transf_space, diagonal_length):
    clusters_transf = []

    
    # Setup coefficients according to paper
    beta_1 = 0.01
    beta_2 = 1.0 / (np.pi * np.pi)
    beta_3 = 4.0 / (diagonal_length * diagonal_length)

    weights = [beta_1, beta_2, beta_2, beta_2, beta_3, beta_3, beta_3, 0, 0]
    
    
    points = [transform_to_point(t) for t in transf_space]
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(points)
    kernel_bandwidth = 1
    #kernel_bandwidth = diagonal_length
    #kernel_bandwidth = estimate_bandwidth(scaled, quantile=0.2)
    #print(kernel_bandwidth)
    msp = MeanShift(bandwidth=kernel_bandwidth, bin_seeding=True)
    msp.kernel = np.diag(weights)
    
    #clusters = msp.fit_predict(scaled)
    tot = msp.fit(scaled)
    clusters = tot.labels_
    centers = tot.cluster_centers_
    centers = scaler.inverse_transform(centers)
    
    for label in np.unique(clusters):
        cluster_transf = [transf_space[i] for i in range(len(clusters)) if clusters[i] == label]
        clusters_transf.append(cluster_transf)
        
        
        
    return clusters_transf, centers