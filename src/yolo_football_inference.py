import torch

# from typing import Generator
# from yolo.src.yolo_utils import *
# from shared.dto import *

# from typing import Generator
# import matplotlib.pyplot as plt
import numpy as np

# from typing import List
import cv2

# import os
# import shutil
# from dataclasses import dataclass
# from onemetric.cv.utils.iou import box_iou_batch
import time
from typing import List, Dict

# from tqdm.notebook import tqdm
# import argparse
# import json
# from shared.config import RedisClient, RedisConfig
# from shared.utils import redis_try_get
# from base64 import b64decode, b64encode
# from genitools import log
from dataclasses import dataclass
from mashumaro import DataClassJSONMixin, DataClassDictMixin
from yolo_utils import load_object_detection_model, generate_frames
from tqdm.notebook import tqdm
import argparse


@dataclass()
class YoloBoundingBox(DataClassJSONMixin, DataClassDictMixin):
    x: int
    y: int
    width: int
    height: int
    class_id: int
    class_name: str
    confidence: float


class YoloFootballInference:
    def __init__(self, video_frame, yolo_model):
        self.video_frame = video_frame
        self.yolo_model = yolo_model
        # if model_type == "ball":
        #     self.yolo_model = load_object_detection_model(
        #         "/project/yolo/models/yolov7_ball", "/project/yolo/models/Ball_Classifier_Trained.pt"
        #     )
        # elif model_type == "player":
        #     self.yolo_model = load_object_detection_model(
        #         "/project/yolo/models/yolov7_player", "/project/yolo/models/Player_Classifier_Pretrained.pt"
        #     )
        # else:
        #     raise Exception("Invalid Model Type!")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device `{self.device}` for inference")

        # # Redis Config
        # redis_config = RedisConfig()
        # self.redis_client = RedisClient(redis_config)
        # self.redis = self.redis_client.conn
        # self.redis = redis

    def yolo_bounding_box_results(
        self, pred: np.ndarray, names: Dict[int, str]
    ) -> List[YoloBoundingBox]:
        result = []
        # print(names)
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            # print(x_min, y_min, class_id, confidence)
            class_id = int(class_id)
            result.append(
                YoloBoundingBox(
                    x=int(x_min),
                    y=int(y_min),
                    width=int(x_max - x_min),
                    height=int(y_max - y_min),
                    class_id=class_id,
                    class_name=names[class_id],
                    confidence=float(confidence),
                )
            )
        return result

    def inference(self):
        """inference function which will detect objects and track it using
         detector model and bytetrack tracking algorithm

        Arguments:
            frame: frame in array
            width: frame size , if width is 720, the the frame size will be
            720x 720
            classes: set class, 'ball' for only detect and track ball and "all"
            for all classcommand: /bin/sh -c '/wait && python /app/pipeline/entrypoint.py'es (player, goalkeeper, ball)
            model: loaded model with our trained detector model
            byte_tracker : ByteTrack instances

        Returns:
           list of result
        """
        # result = None
        # TODO: Has to grab the frame by key
        # print(self, " - Video frame.")
        # print(self.redis_client, " - Redis Client.")
        # print(self.redis, " - Redis.")
        # print(self.ball_detector_yolo_model, " - self.ball_detector_yolo_model.")
        # print(self.player_detector_yolo_model, " - self.player_detector_yolo_model.")
        # print("getting frame from redis")
        # frame = redis_try_get(self.redis, self.video_frame.key)
        # # print("image loaded")
        #
        # # print("converting frame to image")
        # # convert from base64 to real image
        # arr: np.ndarray = np.frombuffer(b64decode(frame), dtype=np.uint8)
        # frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # frame = get from redis with frame_key
        # print("image converted")
        # frame = cv2.imread("/app/sample_input.jpg", cv2.IMREAD_COLOR)

        # print("image send to detector")
        model_detections = self.yolo_model(self.video_frame)
        # player_model_detections = self.player_detector_yolo_model(frame)
        # print("detection completed")
        # post processing detection result
        model_results = self.yolo_bounding_box_results(
            pred=model_detections.pred[0].cpu().numpy(), names=model_detections.names
        )
        return model_results

        # if self.yolo_model == "ball":
        #     result = filter_detections_by_class(detections=model_results, class_name="ball")
        # if self.yolo_model == "player":
        #     result = filter_detections_by_class(detections=model_results, class_name="person")
        #
        # # ball_detections = ball_detections + player_detections
        #
        # return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple example using argparse")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="/app/fifa.mp4",
        help="source of video data for inference",
    )
    parser.add_argument("-W", "--width", type=int, default=1280, help="image width")

    parser.add_argument("-H", "--height", type=int, default=720, help="image height")

    args = parser.parse_args()

    yolo_ball_model = load_object_detection_model(
        "/app/yolov7", "/app/Ball_Classifier_Trained.pt"
    )
    # yolo_player_model = load_object_detection_model(
    #     "/app/yolov7", "/app/Player_Classifier_Pretrained.pt"
    # )

    frame_iterator = iter(
        generate_frames(
            video_file=args.data, resize_width=args.width, resize_height=args.height
        )
    )

    # Count for renaming the frame
    frame_id = 0
    # initiate blank dict to store the result
    result_dict = {}
    # loop over frames
    total_start = time.time()
    for frame in tqdm(
        frame_iterator,
    ):
        # Ball Inference
        ball_inference_start = time.time()
        yolo_ball_inference = YoloFootballInference(frame, yolo_ball_model)
        ball_inference_result = yolo_ball_inference.inference()
        ball_inference_end = time.time()
        print(f"Ball Inference Time: {ball_inference_end - ball_inference_start}")

        # Player Inference
        # player_inference_start = time.time()
        # yolo_player_inference = YoloFootballInference(frame, yolo_player_model)
        # player_inference_result = yolo_player_inference.inference()
        # player_inference_end = time.time()
        # print(f"Player Inference Time: {player_inference_end - player_inference_start}")

    total_end = time.time()
    print(f"Total Time: {total_end - total_start}")
