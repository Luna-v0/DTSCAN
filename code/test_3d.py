"""
3D Point Cloud Clustering with DTSCAN
This script demonstrates DTSCAN on 3D point clouds similar to LiDAR data
Reference: Section 4.2 of Kim & Cho (2019) - Performance Evaluation using 3D Point Clouds
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dtscan import DTSCAN, calculate_psr_vsr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def generate_3d_pedestrians(separation_level=2):
    """
    Generate 3D point clouds representing two pedestrians.
    
    Reference: Section 4.2 - "10 single objects within a detection distance of 15 m 
    were randomly extracted from the data set"
    
    Parameters:
    -----------
    separation_level : int
        1 = far apart (λ=4)
        2 = medium distance (λ=2)  
        3 = very close/touching (λ=1)
    """
    np.random.seed(42)
    
    # Lambda factor for separation (from paper)
    lambda_factors = {1: 4, 2: 2, 3: 1}
    lambda_val = lambda_factors.get(separation_level, 2)
    
    pedestrians = []
    
    # First pedestrian - standing pose
    # Head
    head1 = np.random.randn(50, 3) * 0.1
    head1[:, 2] += 1.7  # Height
    head1[:, 0] -= lambda_val / 2  # Shift left
    
    # Torso
    torso1 = np.random.randn(100, 3)
    torso1[:, 0] *= 0.2
    torso1[:, 1] *= 0.15
    torso1[:, 2] = torso1[:, 2] * 0.3 + 1.0
    torso1[:, 0] -= lambda_val / 2
    
    # Legs
    legs1 = np.random.randn(80, 3)
    legs1[:, 0] *= 0.1
    legs1[:, 1] *= 0.1
    legs1[:, 2] = legs1[:, 2] * 0.4 + 0.3
    legs1[:, 0] -= lambda_val / 2
    
    pedestrian1 = np.vstack([head1, torso1, legs1])
    pedestrians.append(pedestrian1)
    
    # Second pedestrian - walking pose
    # Head
    head2 = np.random.randn(50, 3) * 0.1
    head2[:, 2] += 1.7
    head2[:, 0] += lambda_val / 2  # Shift right
    
    # Torso (slightly rotated)
    torso2 = np.random.randn(100, 3)
    torso2[:, 0] *= 0.2
    torso2[:, 1] *= 0.15
    torso2[:, 2] = torso2[:, 2] * 0.3 + 1.0
    torso2[:, 0] += lambda_val / 2
    torso2[:, 1] += np.random.uniform(-0.1, 0.1, 100)  # Slight rotation
    
    # Legs (one forward, one back for walking)
    leg1 = np.random.randn(40, 3)
    leg1[:, 0] *= 0.1
    leg1[:, 1] = leg1[:, 1] * 0.1 + 0.15  # Forward
    leg1[:, 2] = leg1[:, 2] * 0.4 + 0.3
    leg1[:, 0] += lambda_val / 2
    
    leg2 = np.random.randn(40, 3)
    leg2[:, 0] *= 0.1
    leg2[:, 1] = leg2[:, 1] * 0.1 - 0.15  # Backward
    leg2[:, 2] = leg2[:, 2] * 0.4 + 0.3
    leg2[:, 0] += lambda_val / 2
    
    pedestrian2 = np.vstack([head2, torso2, leg1, leg2])
    pedestrians.append(pedestrian2)
    
    # Add some noise points between pedestrians if they're close
    if separation_level >= 2:
        n_noise = 10 * (4 - separation_level)
        noise = np.random.randn(n_noise, 3)
        noise[:, 0] *= lambda_val / 2
        noise[:, 1] *= 0.3
        noise[:, 2] = noise[:, 2] * 0.5 + 1.0
        pedestrians.append(noise)
    
    # Combine all points
    X = np.vstack(pedestrians)
    
    # Create true labels
    true_labels = np.concatenate([
        np.zeros(len(pedestrian1)),
        np.ones(len(pedestrian2)),
        np.full(len(pedestrians[2]) if len(pedestrians) > 2 else 0, -1)  # Noise
    ])
    
    return X, true_labels


def visualize_3d_clustering(X, labels, title="3D Point Cloud Clustering"):
    """
    Visualize 3D point cloud with cluster labels.
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Original view
    ax1 = fig.add_subplot(121, projection='3d')
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels[unique_labels != -1])))
    
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            mask = labels == label
            ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                       c='black', s=5, alpha=0.3, label='Noise')
        else:
            mask = labels == label
            ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                       c=[colors[color_idx]], s=5, label=f'Person {int(label)+1}')
            color_idx += 1
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title} - 3D View')
    ax1.legend()
    
    # Top-down view (X-Y plane)
    ax2 = fig.add_subplot(122)
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            mask = labels == label
            ax2.scatter(X[mask, 0], X[mask, 1], 
                       c='black', s=5, alpha=0.3, label='Noise')
        else:
            mask = labels == label
            ax2.scatter(X[mask, 0], X[mask, 1], 
                       c=[colors[color_idx]], s=5, label=f'Person {int(label)+1}')
            color_idx += 1
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{title} - Top View (X-Y)')
    ax2.set_aspect('equal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def test_3d_clustering():
    """
    Test DTSCAN on 3D point clouds at different separation levels.
    Reference: Table 4 in the paper
    """
    print("="*80)
    print("3D POINT CLOUD CLUSTERING - PEDESTRIAN SEPARATION")
    print("Reference: Section 4.2 and Table 4 of Kim & Cho (2019)")
    print("="*80)
    
    results_summary = []
    
    for level in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"LEVEL {level} - Separation Factor λ = {[4, 2, 1][level-1]}")
        print(f"{'='*60}")
        
        # Generate 3D pedestrian data
        X, true_labels = generate_3d_pedestrians(separation_level=level)
        print(f"Generated {len(X)} 3D points representing 2 pedestrians")
        
        # Test DTSCAN
        print("\nDTSCAN Results:")
        dtscan = DTSCAN(z_score_threshold=2.0, min_pts=10)  # Higher min_pts for 3D
        labels_dtscan = dtscan.fit_predict(X)
        psr_dtscan, vsr_dtscan = calculate_psr_vsr(true_labels, labels_dtscan)
        n_clusters_dtscan = len(np.unique(labels_dtscan[labels_dtscan != -1]))
        print(f"  Clusters found: {n_clusters_dtscan}")
        print(f"  PSR: {psr_dtscan:.3f}, VSR: {vsr_dtscan:.3f}")
        
        # Test traditional DBSCAN for comparison
        print("\nDBSCAN Results:")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Estimate eps for DBSCAN
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        eps = np.percentile(distances[:, 9], 90)
        
        dbscan = DBSCAN(eps=eps, min_samples=10)
        labels_dbscan = dbscan.fit_predict(X_scaled)
        psr_dbscan, vsr_dbscan = calculate_psr_vsr(true_labels, labels_dbscan)
        n_clusters_dbscan = len(np.unique(labels_dbscan[labels_dbscan != -1]))
        print(f"  Clusters found: {n_clusters_dbscan}")
        print(f"  PSR: {psr_dbscan:.3f}, VSR: {vsr_dbscan:.3f}")
        
        # Store results
        results_summary.append({
            'Level': level,
            'DTSCAN_PSR': psr_dtscan,
            'DTSCAN_VSR': vsr_dtscan,
            'DTSCAN_clusters': n_clusters_dtscan,
            'DBSCAN_PSR': psr_dbscan,
            'DBSCAN_VSR': vsr_dbscan,
            'DBSCAN_clusters': n_clusters_dbscan
        })
        
        # Visualize results
        print("\nVisualizing results...")
        visualize_3d_clustering(X, labels_dtscan, f"DTSCAN - Level {level}")
    
    # Print summary table similar to Table 4 in the paper
    print("\n" + "="*80)
    print("SUMMARY - 3D Point Cloud Clustering Performance")
    print("Reference: Table 4 in Kim & Cho (2019)")
    print("="*80)
    print(f"{'Level':<10} {'Method':<10} {'PSR':<10} {'VSR':<10} {'Clusters':<10}")
    print("-"*50)
    
    for result in results_summary:
        level = result['Level']
        print(f"Level {level:<4} DTSCAN    {result['DTSCAN_PSR']:<10.3f} "
              f"{result['DTSCAN_VSR']:<10.3f} {result['DTSCAN_clusters']:<10}")
        print(f"{'':10} DBSCAN    {result['DBSCAN_PSR']:<10.3f} "
              f"{result['DBSCAN_VSR']:<10.3f} {result['DBSCAN_clusters']:<10}")
        print("-"*50)
    
    print("\n" + "="*80)
    print("KEY OBSERVATIONS (from Section 4.2):")
    print("-"*80)
    print("• DTSCAN maintains better performance as pedestrians get closer")
    print("• Traditional DBSCAN struggles with touching objects (Level 3)")
    print("• VSR remains more stable with DTSCAN across all levels")
    print("• The Delaunay triangulation helps preserve object boundaries")
    print("="*80)


def test_lidar_characteristics():
    """
    Test on data with typical LiDAR characteristics.
    Reference: "clusters representing the shape of objects are irregular and have 
    arbitrary pattern features, which is an inherent limitation of LiDAR"
    """
    print("\n" + "="*80)
    print("TESTING WITH LIDAR-LIKE CHARACTERISTICS")
    print("Reference: Section 4.2 - LiDAR inherent limitations")
    print("="*80)
    
    np.random.seed(42)
    
    # Generate sparse, irregular point cloud typical of LiDAR
    # Object 1: Car-like shape (sparse at distance)
    car_length = 4
    car_width = 2
    car_height = 1.5
    
    # Front and back surfaces (denser)
    n_front = 100
    x_front = np.random.uniform(0, 0.1, n_front)
    y_front = np.random.uniform(-car_width/2, car_width/2, n_front)
    z_front = np.random.uniform(0, car_height, n_front)
    
    n_back = 100
    x_back = np.random.uniform(car_length-0.1, car_length, n_back)
    y_back = np.random.uniform(-car_width/2, car_width/2, n_back)
    z_back = np.random.uniform(0, car_height, n_back)
    
    # Sides (sparser due to angle)
    n_sides = 50
    y_left = np.full(n_sides, -car_width/2)
    x_left = np.random.uniform(0, car_length, n_sides)
    z_left = np.random.uniform(0, car_height, n_sides)
    
    y_right = np.full(n_sides, car_width/2)
    x_right = np.random.uniform(0, car_length, n_sides)
    z_right = np.random.uniform(0, car_height, n_sides)
    
    car1 = np.vstack([
        np.column_stack([x_front, y_front, z_front]),
        np.column_stack([x_back, y_back, z_back]),
        np.column_stack([x_left, y_left, z_left]),
        np.column_stack([x_right, y_right, z_right])
    ])
    
    # Object 2: Another car, adjacent
    car2 = car1.copy()
    car2[:, 0] += car_length + 0.8  # Small gap between cars
    car2[:, 1] += 0.3  # Slight offset
    
    # Add typical LiDAR noise
    ground_noise = np.random.randn(30, 3)
    ground_noise[:, 2] = np.random.uniform(-0.1, 0.1, 30)  # Ground returns
    ground_noise[:, 0] = np.random.uniform(0, 2*car_length+1, 30)
    ground_noise[:, 1] = np.random.uniform(-car_width, car_width, 30)
    
    # Combine
    X = np.vstack([car1, car2, ground_noise])
    true_labels = np.concatenate([
        np.zeros(len(car1)),
        np.ones(len(car2)),
        np.full(len(ground_noise), -1)
    ])
    
    print(f"Generated LiDAR-like data: {len(X)} points")
    print(f"Characteristics: Sparse, irregular density, adjacent objects")
    
    # Test DTSCAN
    print("\nDTSCAN on LiDAR-like data:")
    dtscan = DTSCAN(z_score_threshold=2.5, min_pts=8)
    labels_dtscan = dtscan.fit_predict(X)
    psr, vsr = calculate_psr_vsr(true_labels, labels_dtscan)
    print(f"  Clusters found: {len(np.unique(labels_dtscan[labels_dtscan != -1]))}")
    print(f"  PSR: {psr:.3f}, VSR: {vsr:.3f}")
    
    visualize_3d_clustering(X, labels_dtscan, "DTSCAN on LiDAR-like Data")


if __name__ == "__main__":
    # Test 3D pedestrian clustering at different separation levels
    test_3d_clustering()
    
    # Test on LiDAR-like characteristics
    test_lidar_characteristics()
    
    print("\n" + "="*80)
    print("3D POINT CLOUD TESTING COMPLETE")
    print("="*80)
