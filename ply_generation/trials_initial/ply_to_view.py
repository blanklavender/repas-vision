import open3d as o3d

def load_ply_file(file_path):
    """
    Loads a PLY file and returns an Open3D point cloud object.
    
    Args:
        file_path (str): The path to the PLY file.
        
    Returns:
        o3d.geometry.PointCloud: The loaded point cloud.
    """
    print(f"Loading PLY file: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    
    if not pcd.has_points():
        print("Warning: The point cloud has no points!")
    else:
        print("Successfully loaded the PLY file.")
        print(f"Number of points: {len(pcd.points)}")
        
    return pcd

def visualize_point_cloud(pcd):
    """
    Visualizes the given point cloud using Open3D's visualization tool.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to visualize.
    """
    if pcd is None or not pcd.has_points():
        print("No valid point cloud to display!")
        return
    print("Displaying point cloud...")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Specify the path to your PLY file
    ply_file_path = "./ply_data/mug.ply"  
    
    # Load the point cloud from the PLY file
    point_cloud = load_ply_file(ply_file_path)
    
    # Visualize the loaded point cloud
    visualize_point_cloud(point_cloud)
