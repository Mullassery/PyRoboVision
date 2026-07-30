"""Example: PyRoboVision MCP 2.0 - Autonomous Driving Perception Tools"""

import asyncio
from pyrobovision import PerceptionEngine


async def main():
    # Initialize perception engine with MCP 2.0 support
    engine = PerceptionEngine()

    # Start MCP server on port 8783
    mcp_url = engine.start_mcp_connector(port=8783)
    print(f"✓ PyRoboVision MCP 2.0 running at {mcp_url}")
    print(f"  14 tools available for autonomous driving perception\n")

    # Example tool calls via MCP
    from pyrobovision._mcp_tools import PyRoboVisionMCPHandler

    handler = PyRoboVisionMCPHandler(engine)

    # 1. CLIP embeddings for scene understanding
    print("1. Embedding frames with CLIP...")
    result = await handler.embed_frames_clip(
        num_frames=30,
        model_name="ViT-B/32",
        device="cuda"
    )
    print(f"   ✓ Generated {result['num_frames']} embeddings (dim: {result['embedding_dim']})\n")

    # 2. SAM2 segmentation
    print("2. Segmenting frames with SAM2...")
    result = await handler.segment_frames_sam2(
        num_frames=30,
        model_type="base",
        prompt_type="points"
    )
    print(f"   ✓ Generated {result['masks_generated']} segmentation masks\n")

    # 3. Grounding DINO object detection with text prompts
    print("3. Detecting objects with Grounding DINO...")
    result = await handler.detect_objects_grounding_dino(
        text_prompt="car . truck . pedestrian",
        num_frames=30,
        box_threshold=0.3
    )
    print(f"   ✓ Detected ~{result['detections_per_frame']} objects per frame\n")

    # 4. Multi-camera stitching
    print("4. Stitching panorama from 4 cameras...")
    result = await handler.stitch_panorama(
        num_cameras=4,
        stitch_method="homography",
        overlap_ratio=0.3
    )
    print(f"   ✓ Panorama: {result['panorama_width']}x{result['panorama_height']}\n")

    # 5. BEV projection
    print("5. Projecting to bird's-eye-view...")
    result = await handler.project_bev(
        bev_size=[512, 512],
        camera_height=1.3,
        fov_degrees=90.0
    )
    print(f"   ✓ BEV projection created\n")

    # 6. LiDAR-camera fusion
    print("6. Fusing LiDAR and camera data...")
    result = await handler.fuse_lidar_camera(
        lidar_points=100000,
        camera_detections=20,
        fusion_method="late"
    )
    print(f"   ✓ Fused {result['fused_objects']} 3D objects\n")

    # 7. 3D object detection
    print("7. Detecting 3D objects from multi-modal fusion...")
    result = await handler.detect_3d_objects(
        num_cameras=4,
        lidar_enabled=True,
        radar_enabled=False,
        confidence_threshold=0.5
    )
    print(f"   ✓ Detected {result['detections']} objects\n")

    # 8. Panoptic segmentation
    print("8. Running panoptic segmentation...")
    result = await handler.panoptic_segmentation(
        num_frames=30,
        model_name="panoptic-fpn",
        stuff_classes=11,
        thing_classes=80
    )
    print(f"   ✓ {result['stuff_classes']} stuff + {result['thing_classes']} thing classes\n")

    # 9. Camera calibration
    print("9. Calibrating camera rig...")
    result = await handler.calibrate_cameras(
        num_cameras=4,
        calibration_type="full"
    )
    print(f"   ✓ Calibration error: {result['reprojection_error']:.2f} px\n")

    # 10. Hardware device selection
    print("10. Selecting hardware device...")
    result = await handler.select_hardware_device(
        device_type="cuda",
        memory_gb=12.0,
        batch_size=4
    )
    print(f"   ✓ Using {result['device_name']}\n")

    # 11. Benchmark inference
    print("11. Benchmarking inference performance...")
    result = await handler.benchmark_inference(
        model_name="SAM2",
        input_resolution=[1080, 1920],
        num_frames=100,
        num_runs=5
    )
    print(f"   ✓ Throughput: {result['throughput_fps']:.1f} FPS\n")

    # 12. 3D tracking with Kalman filter
    print("12. Tracking 3D objects...")
    result = await handler.track_objects_kalman(
        num_objects=25,
        max_age=30,
        process_noise=0.01,
        measurement_noise=0.1
    )
    print(f"   ✓ Tracking {result['tracked_objects']} objects\n")

    # 13. Trajectory prediction
    print("13. Predicting object trajectories...")
    result = await handler.predict_trajectory(
        num_objects=25,
        prediction_horizon=30,
        model_type="ctrv",
        use_map_context=True
    )
    print(f"   ✓ Predicted {result['trajectories_predicted']} trajectories\n")

    # 14. Model export
    print("14. Exporting model to ONNX...")
    result = await handler.export_model_onnx(
        model_name="SAM2",
        input_shape=[1, 3, 1080, 1920],
        quantize=True
    )
    print(f"   ✓ Exported: {result['output_file']} ({result['file_size_mb']:.1f} MB)\n")

    print("✓ All 14 MCP 2.0 tools working!")
    print(f"\nMCP 2.0 Endpoint: {mcp_url}")
    print("Port 8783 for PyRoboVision perception tools")

    # Keep server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        engine.stop_mcp_connector()
        print("\n✓ MCP server stopped")


if __name__ == "__main__":
    asyncio.run(main())
