# Dual LiDAR Real Detection & Safety Warning System

A real-time object detection and safety warning system based on dual LiDAR sensors, specifically designed for autonomous driving and intelligent vehicle environmental perception and safety protection.

## 🎯 Core Features7

### 🚗 Intelligent Safety Warning
- **Static Obstacle Detection**: Triggers alerts when static objects (barriers, cones, etc.) are within the lateral safety threshold
- **Dynamic Vehicle Detection**: Detects vehicles ahead and in adjacent lanes, dynamically adjusting safety distances based on relative velocity
- **Oncoming Vehicle Recognition**: Intelligently identifies oncoming vehicles with different safety distance strategies
- **Real-time Risk Assessment**: Physics-based braking distance and reaction time calculations

### 🔍 High-Precision Perception System
- **Dual LiDAR Fusion**: Real-time registration and fusion of two LiDAR sensors for enhanced detection accuracy and coverage
- **SLAM Localization**: Real-time pose estimation and trajectory tracking
- **3D Object Detection**: Precise vehicle and obstacle identification, classification, and bounding box estimation
- **Velocity Estimation**: Multi-frame tracking-based object velocity calculation

## 📊 Technical Highlights

- **Real-time Performance**: Average processing time < 100ms, supports 10Hz real-time processing
- **Intelligent Warning Algorithm**: Dynamic safety distance calculation based on physics braking model
- **Data Persistence**: Automatic saving of triggered event point cloud data for subsequent analysis
- **Visualization Debugging**: Complete processing pipeline visualization tools with single-frame debugging support
- **Flexible Configuration**: Rich parameter configuration for different scenario requirements

## 📁 Project Architecture

```
dual_lidar_real_detection/
├── 📊 Data Acquisition Modules
│   ├── dual_lidar_simulator.py      # Dual LiDAR data simulator (for offline testing)
│   ├── dual_lidar_receiver.py       # Real-time dual LiDAR data receiver
│   └── record_two_lidars_pcap_data.py # Dual LiDAR data recording tool
├── 🧠 Core Processing Modules  
│   ├── slam_processor.py            # SLAM pose estimation processor
│   ├── object_detector.py           # 3D object detector
│   └── safety_analyzer.py           # Safety risk analyzer
├── 🛠️ Utility Modules
│   ├── point_cloud_buffer.py        # Point cloud data buffer management
│   ├── frame_visualizer.py          # Single-frame processing visualization tool
│   └── point_cloud_player.py        # Triggered event playback tool
├── 🚀 Main Programs
│   ├── real_dual_lidar_main.py      # Real-time dual LiDAR main program
│   └── simulate_dual_lidar_main.py  # Simulation data main program
├── 📂 Data Directory
│   └── data/                        # Transformation matrices and configuration files
└── 🎬 Demo Files
    └── Boston_Demo.mov              # System demonstration video
```

## 🚀 Quick Start

### Dependencies

**Python Version**: Python 3.8+

```bash
# Method 1: Using requirements.txt (recommended)
pip install -r requirements.txt

# Method 2: Manual installation of core dependencies
pip install numpy>=1.20.0 open3d>=0.17.0 scikit-learn>=1.0.0
pip install ouster-sdk>=0.10.0 scipy>=1.7.0 pandas>=1.3.0
```

### Installation Steps

1. **Clone Repository**:
```bash
git clone https://github.com/moonmoonmoonmoon/dual_lidar_real_detection.git
cd dual_lidar_real_detection
```

2. **Prepare Data Files**:
```bash
# Ensure data directory contains necessary transformation matrix files
mkdir -p data
# Place ground_transform_target.npy, ground_transform_source.npy, etc.
```

### Usage

#### 🔴 Real-time Mode (Connect to Real LiDAR)
```bash
python real_dual_lidar_main.py
```

#### 🔵 Simulation Mode (Using Recorded Data)
```bash
python simulate_dual_lidar_main.py
```

#### 🎬 Single Frame Debugging
```bash
python frame_visualizer.py <frame_id>
```

#### 📹 Event Playback
```bash
python point_cloud_player.py [trigger_directory]
```

#### 📊 Data Recording
```bash
python record_two_lidars_pcap_data.py
```

## 🎥 System Demo
🖼️ Quick Preview
<div align="center">
  <img src="demo_preview.gif" alt="Dual LiDAR Detection Demo" width="700">
</div>

📥 Full Demo Video
[📹 Download HD Demo Video (71.7MB)](https://github.com/moonmoonmoonmoon/dual_lidar_real_detection/releases/download/v1.0.0/Boston_Demo.mov)

## ⚙️ System Configuration

### LiDAR Hardware Configuration
```python
'lidar1_config': {
    'hostname': 'os-122211001778.local',
    'lidar_port': 7502,
    'imu_port': 7503
},
'lidar2_config': {
    'hostname': 'os-122211001621.local', 
    'lidar_port': 7504,
    'imu_port': 7503
}
```

### Safety Warning Parameters
```python
'safety_config': {
    'method': 'physics',              # Physics braking model
    'min_lateral_distance': 0.3,     # Minimum lateral safety distance (m)
    'min_longitudinal_distance': 1.5, # Minimum longitudinal safety distance (m)
    'reaction_time': 1.0,             # Driver reaction time (s)
    'deceleration': 7.84,             # Maximum braking deceleration (m/s²)
    'ego_half_width': 1.0             # Ego vehicle half-width (m)
}
```

### Detector Parameters
```python
'detector_config': {
    'roi': {
        'length': 20,                 # ROI detection range length (m)
        'width': 6,                   # ROI detection range width (m)
        'height': 4                   # ROI detection range height (m)
    },
    'clustering': {
        'eps': 1.25,                  # Clustering distance threshold
        'min_samples': 80,            # Minimum clustering points
        'voxel_size': 0.05            # Voxel downsampling size
    }
}
```

## 📊 Performance Metrics

- **Detection Range**: 20m forward × 3m left/right
- **Detection Accuracy**: Lateral position error < 10cm, Longitudinal position error < 15cm
- **Processing Latency**: Total processing time < 100ms (10Hz real-time processing)
- **Safety Response**: Physics braking model considering reaction time and braking distance

## 🔧 Hardware Requirements

- **LiDAR Sensors**: Ouster OS-1-128 or compatible models
- **Processor**: Intel i7 or equivalent CPU
- **Memory**: 16GB RAM or more
- **Storage**: SSD with at least 100GB available space
- **Network**: Gigabit Ethernet connection

## 📈 Output Data

The system automatically generates the following data:

1. **Real-time CSV Reports**: `output/lidar_results_YYYYMMDD_HHMMSS.csv`
   - Contains detection results, safety analysis, trigger events for each frame
   
2. **Trigger Event Data**: `output/triggers/trigger_XXXXX/`
   - Point cloud sequence files before and after triggers
   - Detailed information about triggered objects

## 🤝 Contributing

1. Realize Real-time dual LiDAR object detection
2. Create Intelligent safety warning system
3. Complete processing pipeline

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 📧 Contact

- **Project Link**: [https://github.com/moonmoonmoonmoon/dual_lidar_real_detection](https://github.com/moonmoonmoonmoon/dual_lidar_real_detection)

## 🙏 Acknowledgments

- [Ouster SDK](https://github.com/ouster-lidar/ouster_example) - LiDAR data processing
- [Open3D](https://github.com/isl-org/Open3D) - 3D data processing and visualization
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine learning algorithms

---

⭐ If this project helps you, please consider giving it a star!
