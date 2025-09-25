"""
Utils for running evaluation on EgoNN.
"""
import open3d as o3d

from misc.point_clouds import make_open3d_point_cloud, make_open3d_feature

def ransac_fn(query_keypoints, candidate_keypoints, query_features, candidate_features, n_k=128):
    """
    Returns fitness score and estimated transforms
    Estimation using Open3d 6dof ransac based on feature matching.
    """
    kp1 = query_keypoints[:n_k]
    kp2 = candidate_keypoints[:n_k]
    ransac_result = get_ransac_result(
        query_features[:n_k], candidate_features[:n_k], kp1, kp2
    )
    return ransac_result.transformation, len(ransac_result.correspondence_set), ransac_result.fitness

def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000):
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    pcd_coord1 = make_open3d_point_cloud(kp1.numpy())
    pcd_coord2 = make_open3d_point_cloud(kp2.numpy())

    # ransac based eval
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    return ransac_result
