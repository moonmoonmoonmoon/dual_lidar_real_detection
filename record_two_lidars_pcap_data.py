from more_itertools import time_limited
import ouster.sdk.client as client
from ouster.sdk import pcap
from contextlib import closing
from datetime import datetime
import threading
import open3d as o3d
import os
import numpy as np

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def capture_sensor_data(hostname, lidar_port, imu_port, n_seconds):
    try:
        # 配置传感器
        config = client.SensorConfig()
        config.udp_port_lidar = lidar_port
        config.udp_port_imu = imu_port

        # 使用新的SensorPacketSource API连接传感器
        with closing(client.SensorPacketSource([(hostname, config)],
                                               buf_size=640).single_source(0)) as source:

            # 创建文件名
            time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
            meta = source.metadata
            fname_base = f"{meta.prod_line}_{meta.sn}_{meta.config.lidar_mode}_{time_part}"

            # 保存元数据
            print(f"Saving sensor metadata to: {fname_base}.json")
            with open(f"{fname_base}.json", "w") as f:
                f.write(source.metadata.to_json_string())

            # 记录数据包
            print(f"Writing to: {fname_base}.pcap (Ctrl-C to stop early)")
            source_it = time_limited(n_seconds, source)
            n_packets = pcap.record(source_it, f"{fname_base}.pcap")

            print(f"Captured {n_packets} packets from sensor {hostname}")

    except Exception as e:
        print(f"Failed to capture data from sensor {hostname}: {e}")


def main():
    n_seconds = 6  # 捕获持续时间(秒)
    output_dir = "sensor_data"  # 输出目录

    # 传感器配置信息
    sensor_configs = [
        {
            'hostname': '192.168.2.202',  # 1621传感器
            # 'hostname': '169.254.134.174',
            'lidar_port': 7504,
            'imu_port': 7505
        },
        {
            'hostname': '192.168.1.201',  # 1778传感器
            'lidar_port': 7502,
            'imu_port': 7503
        }
    ]

    # 确保输出目录存在
    ensure_dir(output_dir)

    # 创建线程列表
    threads = []

    # 为每个传感器创建并启动线程
    for config in sensor_configs:
        thread = threading.Thread(
            target=capture_sensor_data,
            args=(
                config['hostname'],
                config['lidar_port'],
                config['imu_port'],
                n_seconds
                # output_dir
            )
        )
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()