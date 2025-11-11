# Save as run_custom.py
from estimater import *
from custom_reader import CustomDataReader
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, required=True, 
                       help='Path to CAD model (.obj or .ply)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to your data directory')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default='./output')
    args = parser.parse_args()
    
    set_logging_format()
    set_seed(0)
    
    # Load mesh
    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    
    # Setup output directories
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
    
    # Get bounding box
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    # Initialize FoundationPose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, 
                         mesh=mesh, scorer=scorer, refiner=refiner, 
                         debug_dir=debug_dir, debug=debug, glctx=glctx)
    
    logging.info("Estimator initialization done")
    
    # Load your custom data
    reader = CustomDataReader(data_dir=args.data_dir)
    
    # Process frames
    for i in range(len(reader)):
        logging.info(f'Processing frame {i}/{len(reader)}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        
        if i == 0:
            # First frame: register object
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, 
                              ob_mask=mask, iteration=args.est_refine_iter)
        else:
            # Subsequent frames: track object
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, 
                               iteration=args.track_refine_iter)
        
        # Save pose
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))
        
        # Visualize
        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, 
                              K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        
        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
    
    logging.info(f"Done! Results saved to {debug_dir}")