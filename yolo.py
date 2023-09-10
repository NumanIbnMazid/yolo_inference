import torch
from typing import Generator
from yolo_utils import *
from typing import Generator
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import cv2
import os
import shutil
from dataclasses import dataclass
# from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
import time
from typing import List
from tqdm.notebook import tqdm
import argparse
import json


# def inference(frame, width, height, classes, model, byte_tracker, data, frame_id):
def inference(args):

    """inference function which will detect objects and track it using
     detector model and bytetrack tracking algorithm

    Arguments:
        frame: frame in array
        width: frame size , if width is 720, the the frame size will be
        720x 720
        classes: set class, 'ball' for only detect and track ball and "all"
        for all classes (player, goalkeeper, ball)
        model: loaded model with our trained detector model
        byte_tracker : ByteTrack instances

    Returns:
       list of result
    """
    # run object detector

    ball_detector_yolo_model = load_object_detection_model(args.repo, "path to model")
    player_detector_yolo_model = load_object_detection_model(args.repo, "path to model")

    frame_key = args.video_frame.frame_key

    frame= None

    # frame = get from redis with frame_key

    ball_model_detections = ball_detector_yolo_model(frame)

    player__model_detections  = player_detector_yolo_model(frame)


    # results = model(frame, )
    # print(results.print())

    # post processing detection result
    ball_model_results = YoloBoundingBox.from_results(
        pred=ball_model_detections.pred[0].cpu().numpy(),
        names=ball_detector_yolo_model.names)

    player_model_results = YoloBoundingBox.from_results(
        pred=player__model_detections.pred[0].cpu().numpy(),
        names=player__model_detections.names)

    ball_detections = filter_detections_by_class(detections=ball_model_results,class_name="ball")
    player_detections = filter_detections_by_class(detections=player_model_results, class_name="person")

    ball_detections = ball_detections+player_detections

    return ball_detections



    # print(detections)

    # get detection reult filter by class
    # if (args.classes == 'all'):
    #     goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
    #     player_detections = filter_detections_by_class(detections=detections, class_name="player")
    #     ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
    #     referee_detections = filter_detections_by_class(detections=detections, class_name="referee")
    #
    #     detections = player_detections + goalkeeper_detections + ball_detections + referee_detections
    #
    # elif (args.classes == 'ball'):
    #     ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
    #     detections = ball_detections
    # elif (args.classes == 'pretrained'):
    #     # ball_detections = filter_detections_by_class(detections=detections, class_name="sports ball")
    #     person_detections = filter_detections_by_class(detections=detections, class_name="person")
    #
    #     detections = person_detections
    # print(detections)
    # # initiate tracker
    # byte_tracker = BYTETracker(BYTETrackerArgs())
    # Now Track player in a single frame

    # if len(detections) > 0:
    #     # print(byte_tracker.tracked_stracks)
    #
    #     tracks = byte_tracker.update(
    #         output_results=detections2boxes(detections=detections),
    #         img_info=frame.shape,
    #         img_size=frame.shape,
    #         # A=A,B=B,C=C,
    #     )
    #     # print('tracks####',tracks)
    # else:
    #     # if there is no detection, so there is no traking result
    #     tracks = []
    #     track_final_result = []
    #
    #     # post-processing track results
    # if len(tracks) != 0:
    #
    #     track_final_result = match_detections_with_tracks(detections=detections, tracks=tracks)
    # # print(type(track_final_result[0]))
    # #  print(track_final_result)
    # # print(track_final_result)
    # else:
    #     track_final_result = []
    # # if (args.save == 1):
    # #     annotated_image = frame.copy()
    # #     annotated_image = text_annotator.annotate(
    # #        image=annotated_image,
    # #        detections=track_final_result)
    #
    # # cv2.imwrite( '/root/projects/tracking/bytetrack/frames_result/'+ 'infer_2_'+ str(frame_id)+".jpg", annotated_image)
    #
    # # print(type(track_final_result))
    # # print(track_final_result)
    # return track_final_result


def process_result(result_list):
    """Process the result list into a dictionary

    Arguments:
            result_list: A list of Detection object

    Returns:
       list of result
    """

    detection_dict = {}
    # create object ID
    count = 1
    for result_object in result_list:
        # print(result_object)
        # frame_dict[str(frame_id)] = ''
        result_object_dict = {
            "x": result_object.rect.x,
            "y": result_object.rect.y,
            "width": result_object.rect.width,
            "height": result_object.rect.height,
            "class_id": result_object.class_id,
            "class_name": result_object.class_name,
            "confidence": result_object.confidence,
            "tracker_id": result_object.tracker_id
        }
        detection_dict['object_' + str(count)] = result_object_dict
        count = count + 1

    return detection_dict


# convert result dictionary to json
def convert_dict_to_json(result_dict, output):
    # json_string = json.dumps(result_dict)
    # save JSON string to a file
    with open(output, 'w') as f:
        json.dump(result_dict, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A simple example using argparse')

    parser.add_argument('-n', '--name', type=str, default='World',
                        help='The name to greet')

    parser.add_argument('-M', '--mode', type=str, default='video',
                        help='video or image')

    parser.add_argument('-fps', '--fps', type=int, default=30,
                        help='Number of times to greet')

    parser.add_argument('-W', '--width', type=int, default=1280,
                        help='image width')

    parser.add_argument('-H', '--height', type=int, default=720,
                        help='image height')

    parser.add_argument('-d', '--data', type=str, default='/root/projects/tracking/bytetrack/data/15_min_splited.mp4',
                        help='source of video data for inference')

    parser.add_argument('-o', '--output', type=str,
                        default='/root/projects/tracking/bytetrack/ByteTrack/output_json/15_min_splited.json',
                        help='source of video data for inference')

    parser.add_argument('-r', '--repo', type=str, default='/root/projects/tracking/bytetrack/yolov7',
                        help='source of yolov7 repo')

    parser.add_argument('-w', '--weights', type=str,
                        default='/root/projects/FIFA_YOLO/yolov7/runs/train/1280x720_All_Class/weights/last.pt',
                        # /root/projects/FIFA_YOLO/yolov7/runs/train/v1_300_9_no_960/weights/best.pt
                        help='source of trained weights')

    parser.add_argument('-c', '--classes', type=str, default='all',
                        help='all for player, referee, goalkeeper and ball ')

    parser.add_argument('-s', '--save', type=int, default=1,
                        help='save the frame and video')

    args = parser.parse_args()

    inference(args)

    # load model
    # model = load_object_detection_model(args.repo, args.weights)
    # print(model.names)
    # initiate tracker
    # byte_tracker = BYTETracker(BYTETrackerArgs(), )
    # print(byte_tracker.a)

    # text_annotator = TextAnnotator(background_color=Color(255, 255, 255), text_color=Color(0, 0, 0), text_thickness=2)

    # if args.mode == 'video':
    #     # start=time.time()
    #     # get fresh video frame generator
    #     frame_iterator = iter(generate_frames(video_file=args.data, resize_width=args.width, resize_height=args.height))
    #
    #     # Count for renaming the frame
    #     frame_id = 0
    #     # initiate blank dict to store the result
    #     result_dict = {}
    #     # loop over frames
    #     start_final = time.time()
    #     for frame in tqdm(frame_iterator, ):
    #         # get the inference result (as list)
    #         #    start=time.time()
    #         # Create frame identity number
    #         frame_id = frame_id + 1
    #         result_list = inference(frame, args.width, args.height, args.classes, model, byte_tracker, args.data,
    #                                 frame_id)
    #         #    if (len(A)>0):
    #         #     print('a:',(A[0]._tlwh))
    #         #     print('a:',(A[0].score))
    #         #    end=time.time()
    #         #    print("Detect and track Single Frame:" + str(end-start)+ " seconds")
    #
    #         # process the result list to dict for single frame inference
    #         detection_dict = process_result(result_list)
    #         # push the dict into the final result dict
    #         result_dict['frame_' + str(frame_id)] = detection_dict
    #
    #     # convert dictionary to JSON format
    #     if (args.save == 1):
    #         convert_dict_to_json(result_dict, args.output)
    #
    #     time_taken = round((time.time() - start_final), 3)
    #     print("Time for Inference:", time_taken)

    # elif (args.mode == 'image'):
    #     # load image
    #     img = cv2.imread(args.data, cv2.IMREAD_COLOR)
    #     # resize the frame
    #     frame = cv2.resize(img, (args.width, args.height))
    #
    #     # start=time.time()
    #
    #     result_list = inference(frame, args.width, args.height, args.classes, model, byte_tracker, args.data, 1)
    #
    #     # end=time.time()
    #
    #     # process the result list to dict for single frame inference
    #     detection_dict = process_result(result_list)
    #     # push the dict into the final result dict
    #     result_dict = {}
    #     result_dict['1'] = detection_dict
    #     # convert dictionary to JSON format
    #     if (args.save == 1):
    #         convert_dict_to_json(result_dict, args.output)
    #     # print("Detect and track Single Frame:" + str(end-start)+ " seconds")
    #     # process_result(result_list)

