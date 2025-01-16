from rclpy.serialization import deserialize_message, serialize_message

import os
import yaml
import rosbag2_py

import cv2
from cv_bridge import CvBridge
import csv
import math

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from novatel_oem7_msgs.msg import BESTPOS, INSPVA
from autoware_auto_vehicle_msgs.msg import VelocityReport
from tf2_msgs.msg import TFMessage

import lanelet2.core
import lanelet2.io
import lanelet2.projection
    
def read_camera_info_from_file(camera_info_path):
    with open(camera_info_path, 'r') as f:
        cam_info_data = yaml.safe_load(f)
    cam_info_msg = CameraInfo()
    cam_info_msg.header.frame_id = 'v4l_frame'
    cam_info_msg.width = cam_info_data["image_width"]
    cam_info_msg.height = cam_info_data["image_height"]
    cam_info_msg.distortion_model = cam_info_data["distortion_model"]

    cam_info_msg.d = cam_info_data["D"]
    cam_info_msg.k = cam_info_data["K"]
    cam_info_msg.r = cam_info_data["R"]
    cam_info_msg.p = cam_info_data["P"]
    return cam_info_msg

def convert_bestpos_to_groundtruthpose(bestpos_msg, projector):
    latitude = bestpos_msg.lat
    longitude = bestpos_msg.lon
    altitude = bestpos_msg.hgt + bestpos_msg.undulation
    gps_point = lanelet2.core.GPSPoint(latitude, longitude, altitude)
    local_point = projector.forward(gps_point)

    pose_msg = PoseStamped()
    pose_msg.header = bestpos_msg.header
    pose_msg.header.frame_id = "map"
    pose_msg.pose.position.x = local_point.x
    pose_msg.pose.position.y = local_point.y
    pose_msg.pose.position.z = altitude
    return pose_msg

def convert_inspva_to_velocity(inspva_msg):
    north_velocity = inspva_msg.north_velocity 
    east_velocity = inspva_msg.east_velocity
    horizontal_speed = math.sqrt(north_velocity**2 + east_velocity**2)

    # 發布 VelocityReport
    velocity_report = VelocityReport()
    velocity_report.header.stamp = inspva_msg.header.stamp  # 使用 INSPVA 訊息的時間戳
    velocity_report.header.frame_id = 'base_link'
    velocity_report.longitudinal_velocity = horizontal_speed
    
    return velocity_report

def load_tf_static(tf_static_path):
    with open(tf_static_path, 'r') as f:
        tf_static_data = yaml.safe_load(f)

    tf_msg = TFMessage()
    static_transforms = []

    for transform in tf_static_data['static_transforms']:
        tf = TransformStamped()
        tf.header.frame_id = transform['header']['frame_id']
        tf.child_frame_id = transform['child_frame_id']
        
        tf.transform.translation.x = transform['transform']['translation']['x']
        tf.transform.translation.y = transform['transform']['translation']['y']
        tf.transform.translation.z = transform['transform']['translation']['z']

        tf.transform.rotation.x = transform['transform']['rotation']['x']
        tf.transform.rotation.y = transform['transform']['rotation']['y']
        tf.transform.rotation.z = transform['transform']['rotation']['z']
        tf.transform.rotation.w = transform['transform']['rotation']['w']

        static_transforms.append(tf)
    
    tf_msg.transforms = static_transforms

    return tf_msg

def convert_rosbag2(config_path: str):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    read_bag_path = config["read_bag_path"]
    write_bag_path = config["write_bag_path"]
    groundtruth_path = config["groundtruth_path"]
    imu_topic = config.get("imu_topic", "/imu/data")
    gps_topic = config.get("gps_topic", "/novatel/oem7/bestpos")
    inspva_topic = config.get("inspva_topic", "/novatel/oem7/inspva")
    camera_list = config.get("camera_list", [])
    tf_static_path = config.get("tf_static_path", "tf_static.yaml")
    
    csv_file = open(groundtruth_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["sec", "nsec", "x", "y"])

    bridge = CvBridge()


    # 2. Lanelet2 用於 BESTPOS -> local map
    map_origin = config["map_origin"]
    lat = map_origin["latitude"]
    lon = map_origin["longitude"]
    elev = map_origin["elevation"]
    position = lanelet2.core.GPSPoint(lat, lon, elev)
    origin = lanelet2.io.Origin(position)
    projector = lanelet2.projection.UtmProjector(origin)

    # 3. 準備 rosbag2 讀取
    if not os.path.exists(read_bag_path):
        print(f"[Error] Read bag not found: {read_bag_path}")
        return

    # 4. 準備 rosbag2 寫出
    if os.path.exists(write_bag_path):
        print(f"[Warning] Output bag already exists, it will be overwritten: {write_bag_path}")
        os.remove(write_bag_path)


    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=read_bag_path,
        storage_id='sqlite3'  
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {topic_info.name: topic_info.type for topic_info in topic_types}

    writer = rosbag2_py.SequentialWriter()
    out_storage_options = rosbag2_py.StorageOptions(
        uri=write_bag_path,
        storage_id='sqlite3'
    )
    out_converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer.open(out_storage_options, out_converter_options)

    # 1) camera_list
    for cam in camera_list:
        camera_name = cam["camera_name"]
        camera_info_topic = cam["camera_info_topic"]
        image_topic = cam["image_topic"]
        if image_topic in type_map:

            topic = image_topic + "/compressed"
            type_name = "sensor_msgs/msg/CompressedImage"

            writer.create_topic(
                rosbag2_py._storage.TopicMetadata(
                    name=topic,
                    type=type_name,
                    serialization_format='cdr'
                )
            )

            topic = camera_info_topic
            type_name = "sensor_msgs/msg/CameraInfo"
                
            writer.create_topic(
                rosbag2_py._storage.TopicMetadata(
                    name=topic,
                    type=type_name,
                    serialization_format='cdr'
                )
            )  
    
    # 2) /groundtruth_pose (PoseStamped)
    groundtruthpose_topic_metadata = rosbag2_py.TopicMetadata(
        name="/groundtruth_pose",
        type="geometry_msgs/msg/PoseStamped",
        serialization_format="cdr"
    )
    writer.create_topic(groundtruthpose_topic_metadata)

    # 3) /initialpose (PoseWithCovarianceStamped)
    initialpose_qos_yaml = """- history: 3\n  depth: 0\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 0\n    nsec: 0\n  lifespan:\n    sec: 0\n    nsec: 0\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 0\n    nsec: 0\n  avoid_ros_namespace_conventions: false """
    writer.create_topic(
        rosbag2_py._storage.TopicMetadata(
            name="/initialpose",
            type="geometry_msgs/msg/PoseWithCovarianceStamped",
            serialization_format='cdr',
            offered_qos_profiles=initialpose_qos_yaml
        )
    )

    # 4) /vehicle/status/velocity_status (VelocityReport)
    writer.create_topic(
        rosbag2_py._storage.TopicMetadata(
            name="/vehicle/status/velocity_status",
            type="autoware_auto_vehicle_msgs/msg/VelocityReport",
            serialization_format='cdr'
        )
    )

    # 5) /tf_static (TFMessage)
    tf_static_qos_yaml = """- history: 3\n  depth: 0\n  reliability: 1\n  durability: 1\n  deadline:\n    sec: 0\n    nsec: 0\n  lifespan:\n    sec: 0\n    nsec: 0\n  liveliness: 1\n  liveliness_lease_duration:\n    sec: 0\n    nsec: 0\n  avoid_ros_namespace_conventions: false """
    writer.create_topic(
        rosbag2_py._storage.TopicMetadata(
            name="/tf_static",
            type="tf2_msgs/msg/TFMessage",
            serialization_format='cdr',
            offered_qos_profiles=tf_static_qos_yaml
        )
    )

    # 6) /imu (sensor_msgs/msg/Imu)
    writer.create_topic(
        rosbag2_py._storage.TopicMetadata(
            name= imu_topic,
            type= type_map[imu_topic],  
            serialization_format='cdr'
        )
    )

    first_gps_written = False
    first_tf_msg = None
    tf_timestamp = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        # (A) Camera topics
        for cam in camera_list:
            camera_name = cam["camera_name"]
            image_topic = cam["image_topic"]
            camera_info_topic = cam["camera_info_topic"]
            camera_info_path = cam["camera_info_path"]

            if topic == image_topic:
                # 1) Image
                try:
                    image_msg = deserialize_message(data, Image)
                    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
                    success, compressed_image = cv2.imencode(".jpg", cv_image)
                    if success:
                        compressed_msg = CompressedImage()
                        compressed_msg.header = image_msg.header
                        compressed_msg.format = "jpeg"
                        compressed_msg.data = compressed_image.tobytes()

                        compressed_topic = f"{image_topic}/compressed"
                        writer.write(
                            compressed_topic,
                            serialize_message(compressed_msg),
                            t
                        ) 

                except Exception as e:
                    print(f"Error compressing image: {e}")

                # 2) CameraInfo
                camerainfo_msg = read_camera_info_from_file(camera_info_path)
                writer.write(camera_info_topic, serialize_message(camerainfo_msg), t)

        # 額外：若是 BESTPOS => groundtruth_pose
        if topic == gps_topic:
            try:
                msg = deserialize_message(data, BESTPOS)
                groundtruthpose_msg = convert_bestpos_to_groundtruthpose(msg, projector)
                writer.write("/groundtruth_pose", serialize_message(groundtruthpose_msg), t)
                csv_writer.writerow([msg.header.stamp.sec, msg.header.stamp.nanosec, groundtruthpose_msg.pose.position.x, groundtruthpose_msg.pose.position.y])
                
                if not first_gps_written:
                    initialpose_msg = PoseWithCovarianceStamped()
                    initialpose_msg.header.frame_id = groundtruthpose_msg.header.frame_id
                    initialpose_msg.pose.pose.position = groundtruthpose_msg.pose.position
                    initialpose_msg.pose.pose.orientation.z = 1.0
                    initialpose_msg.pose.pose.orientation.w = 0.0
                    initialpose_msg.pose.pose.position.z = 20.0
                    initialpose_msg.header.frame_id = "map"
                    writer.write("/initialpose", serialize_message(initialpose_msg), t)
                    first_gps_written = True
                    tf_timestamp = t

            except Exception as e:
                print(f"Error processing BESTPOS: {e}")

        elif topic == inspva_topic:
            try:
                msg = deserialize_message(data, INSPVA)
                velocity_report = convert_inspva_to_velocity(msg)
                writer.write("/vehicle/status/velocity_status", serialize_message(velocity_report), t)
            except Exception as e:
                print(f"Error processing INSPVA: {e}")

        elif topic == imu_topic:
            writer.write(topic, data, t)



    # 5. After exiting the loop, write /tf_static
    tf_static_msg = load_tf_static(tf_static_path)
    if tf_static_msg and tf_timestamp:
        writer.write(
            "/tf_static",
            serialize_message(tf_static_msg),           
            tf_timestamp
        )

    csv_file.close()
    print(f"Processing completed! Output bag: {write_bag_path}")

def main():
    config_path = os.path.dirname(os.path.realpath(__file__))+"/config/" + "config.yaml"
    convert_rosbag2(config_path)

if __name__ == "__main__":
    main()
