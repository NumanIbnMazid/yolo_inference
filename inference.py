from header import *
from main_copy import *
import argparse
from annotator import *
import json


def inference(frame, width, height, classes, model, byte_tracker, data, frame_id):
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
    results = model(
        frame,
    )
    # print(results.print())

    # post processing detection result
    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(), names=model.names
    )
    # print(detections)

    # get detection reult filter by class
    if classes == "all":
        goalkeeper_detections = filter_detections_by_class(
            detections=detections, class_name="goalkeeper"
        )
        player_detections = filter_detections_by_class(
            detections=detections, class_name="player"
        )
        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )
        referee_detections = filter_detections_by_class(
            detections=detections, class_name="referee"
        )

        detections = (
            player_detections
            + goalkeeper_detections
            + ball_detections
            + referee_detections
        )

    elif classes == "ball":
        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )
        detections = ball_detections
    elif classes == "pretrained":
        # ball_detections = filter_detections_by_class(detections=detections, class_name="sports ball")
        person_detections = filter_detections_by_class(
            detections=detections, class_name="person"
        )

        detections = person_detections
    # print(detections)
    # # initiate tracker
    # byte_tracker = BYTETracker(BYTETrackerArgs())
    # Now Track player in a single frame
    if len(detections) > 0:
        # print(byte_tracker.tracked_stracks)

        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape,
            # A=A,B=B,C=C,
        )
        # print('tracks####',tracks)
    else:
        # if there is no detection, so there is no traking result
        tracks = []
        track_final_result = []

        # post-processing track results
    if len(tracks) != 0:
        track_final_result = match_detections_with_tracks(
            detections=detections, tracks=tracks
        )
    # print(type(track_final_result[0]))
    #  print(track_final_result)
    # print(track_final_result)
    else:
        track_final_result = []
    # if (args.save == 1):
    #     annotated_image = frame.copy()
    #     annotated_image = text_annotator.annotate(
    #        image=annotated_image,
    #        detections=track_final_result)

    # cv2.imwrite( '/root/projects/tracking/bytetrack/frames_result/'+ 'infer_2_'+ str(frame_id)+".jpg", annotated_image)

    # print(type(track_final_result))
    # print(track_final_result)
    return track_final_result


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
            #         "detection_id": count,
            "rect": {
                "x": result_object.rect.x,
                "y": result_object.rect.y,
                "width": result_object.rect.width,
                "height": result_object.rect.height,
            },
            "class_id": result_object.class_id,
            "class_name": result_object.class_name,
            "confidence": result_object.confidence,
            "tracker_id": result_object.tracker_id,
        }
        detection_dict["object_" + str(count)] = result_object_dict
        count = count + 1

    return detection_dict


# convert result dictionary to json
def convert_dict_to_json(result_dict, output):
    # json_string = json.dumps(result_dict)
    # save JSON string to a file
    with open(output, "w") as f:
        json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device} for Inference")
    parser = argparse.ArgumentParser(description="A simple example using argparse")

    parser.add_argument(
        "-n", "--name", type=str, default="World", help="The name to greet"
    )

    parser.add_argument(
        "-M", "--mode", type=str, default="video", help="video or image"
    )

    parser.add_argument(
        "-fps", "--fps", type=int, default=30, help="Number of times to greet"
    )

    parser.add_argument("-W", "--width", type=int, default=1280, help="image width")

    parser.add_argument("-H", "--height", type=int, default=720, help="image height")

    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="/app/fifa.mp4",
        help="source of video data for inference",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="/app/result.json",
        help="source of video data for inference",
    )

    parser.add_argument(
        "-r",
        "--repo",
        type=str,
        default="/app/yolov7",
        help="source of yolov7 repo",
    )

    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="/app/yolov7.pt",
        help="source of trained weights",
    )

    parser.add_argument(
        "-c",
        "--classes",
        type=str,
        default="pretrained",
        help="all for player, referee, goalkeeper and ball ",
    )

    parser.add_argument(
        "-s", "--save", type=int, default=1, help="save the frame and video"
    )

    args = parser.parse_args()

    # load model
    model = load_object_detection_model(args.repo, args.weights)
    # print(model.names)
    # initiate tracker
    byte_tracker = BYTETracker(
        BYTETrackerArgs(),
    )
    # print(byte_tracker.a)

    # text_annotator = TextAnnotator(
    #     background_color=Color(255, 255, 255),
    #     text_color=Color(0, 0, 0),
    #     text_thickness=2,
    # )

    if args.mode == "video":
        # start=time.time()
        # get fresh video frame generator
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
        start_final = time.time()
        for frame in tqdm(
            frame_iterator,
        ):
            # get the inference result (as list)
            #    start=time.time()
            # Create frame identity number
            frame_id = frame_id + 1
            start_time = time.time()
            result_list = inference(
                frame,
                args.width,
                args.height,
                args.classes,
                model,
                byte_tracker,
                args.data,
                frame_id,
            )
            end_time = time.time()
            # print("Inference Time (Per frame):" + str(end_time-start_time) + " seconds")
            
            #    if (len(A)>0):
            #     print('a:',(A[0]._tlwh))
            #     print('a:',(A[0].score))
            #    end=time.time()
            #    print("Detect and track Single Frame:" + str(end-start)+ " seconds")

            # process the result list to dict for single frame inference
            detection_dict = process_result(result_list)
            # push the dict into the final result dict
            result_dict["frame_" + str(frame_id)] = detection_dict

        # convert dictionary to JSON format
        if args.save == 1:
            convert_dict_to_json(result_dict, args.output)

        time_taken = round((time.time() - start_final), 3)
        print("Time for Inference:", time_taken)

    elif args.mode == "image":
        # load image
        img = cv2.imread(args.data, cv2.IMREAD_COLOR)
        # resize the frame
        frame = cv2.resize(img, (args.width, args.height))

        # start=time.time()

        result_list = inference(
            frame,
            args.width,
            args.height,
            args.classes,
            model,
            byte_tracker,
            args.data,
            1,
        )

        # end=time.time()

        # process the result list to dict for single frame inference
        detection_dict = process_result(result_list)
        # push the dict into the final result dict
        result_dict = {}
        result_dict["1"] = detection_dict
        # convert dictionary to JSON format
        if args.save == 1:
            convert_dict_to_json(result_dict, args.output)
        # print("Detect and track Single Frame:" + str(end-start)+ " seconds")
        # process_result(result_list)
