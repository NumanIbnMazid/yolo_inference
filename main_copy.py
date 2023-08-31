from header import *


def generate_frames(
    video_file: str, resize_width: int, resize_height: int
) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        start = time.time()
        success, frame = video.read()

        # if not success:
        #     break

        # yield frame

        # video.release()

        if success:
            # cv2.imwrite( '/root/projects/tracking/bytetrack/'+'frame'+".jpg", frame)
            resized_frame = cv2.resize(frame, (resize_width, resize_height))
            # cv2.imwrite( '/root/projects/tracking/bytetrack/'+'resized'+".jpg", resized_frame)
            # print(resized_frame.shape)
            end = time.time()
            # print("frame itereator_###:" + str(end - start) + " seconds")

            yield resized_frame

        else:
            break
    video.release()


def load_object_detection_model(repo_path, weights_path):
    """load trained object detection model weight

    Arguments:
        repo_path (str): repo directory path of object detector , i.e. '/root/projects/tracking/bytetrack/yolov7'
        weights_path (str): load pretrained weights into the model

    Returns:
        models.common.Detections object
    """
    start = time.time()

    model = torch.hub.load(
        repo_path, "custom", weights_path, source="local", force_reload=True
    )  # local repo
    end = time.time()
    # print("load_object_detection_model:" + str(end-start)+ " seconds")

    return model


# Video Config

from dataclasses import dataclass

import cv2


"""
usage example:

video_config = VideoConfig(
    fps=30, 
    width=1920, 
    height=1080)
video_writer = get_video_writer(
    target_video_path=TARGET_VIDEO_PATH, 
    video_config=video_config)

for frame in frames:
    ...
    video_writer.write(frame)
    
video_writer.release()
"""


# stores information about output video file, width and height of the frame must be equal to input video
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int


# create cv2.VideoWriter object that we can use to save output video
def get_video_writer(
    target_video_path: str, video_config: VideoConfig
) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    # print(target_video_path)
    return cv2.VideoWriter(
        target_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=video_config.fps,
        frameSize=(video_config.width, video_config.height),
        isColor=True,
    )


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


"""
BYTETracker does not assign tracker_id to existing bounding boxes but rather
predicts the next bounding box position based on previous one. Therefore, we 
need to find a way to match our bounding boxes with predictions.

usage example:

byte_tracker = BYTETracker(BYTETrackerArgs())
for frame in frames:
    ...
    results = model(frame, size=1280)
    detections = Detection.from_results(
        pred=results.pred[0].cpu().numpy(), 
        names=model.names)
    ...
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape)
    detections = match_detections_with_tracks(detections=detections, tracks=tracks)
"""


# converts List[Detection] into format that can be consumed by match_detections_with_tracks function
def detections2boxes(
    detections: List[Detection], with_confidence: bool = True
) -> np.ndarray:
    return np.array(
        [
            [
                detection.rect.top_left.x,
                detection.rect.top_left.y,
                detection.rect.bottom_right.x,
                detection.rect.bottom_right.y,
                detection.confidence,
            ]
            if with_confidence
            else [
                detection.rect.top_left.x,
                detection.rect.top_left.y,
                detection.rect.bottom_right.x,
                detection.rect.bottom_right.y,
            ]
            for detection in detections
        ],
        dtype=float,
    )


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: List[Detection], tracks: List[STrack]
) -> List[Detection]:
    detection_boxes = detections2boxes(detections=detections, with_confidence=False)
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            detections[detection_index].tracker_id = tracks[tracker_index].track_id
    return detections
