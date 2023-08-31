from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import cv2

import numpy as np


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding
        )

    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y


# detection utilities


@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None

    @classmethod
    def from_results(cls, pred: np.ndarray, names: Dict[int, str]) -> List[Detection]:
        result = []
        # print(pred.shape)
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id = int(class_id)
            result.append(Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence)
            ))
        return result


def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [
        detection
        for detection
        in detections
        if detection.class_name == class_name
    ]


# draw utilities


# @dataclass(frozen=True)
# class Color:
#     r: int
#     g: int
#     b: int
#
#     @property
#     def bgr_tuple(self) -> Tuple[int, int, int]:
#         return self.b, self.g, self.r
#
#     @classmethod
#     def from_hex_string(cls, hex_string: str) -> Color:
#         r, g, b = tuple(int(hex_string[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
#         return Color(r=r, g=g, b=b)


# def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
#     cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
#     return image
#
#
# def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
#     cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
#     return image


# def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
#     cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
#     return image
#
#
# def draw_filled_polygon(image: np.ndarray, countour: np.ndarray, color: Color) -> np.ndarray:
#     cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
#     return image
#
#
# def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2) -> np.ndarray:
#     cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.bgr_tuple, thickness, 2, False)
#     return image


# def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
#     cv2.ellipse(
#         image,
#         center=rect.bottom_center.int_xy_tuple,
#         axes=(int(rect.width), int(0.35 * rect.width)),
#         angle=0.0,
#         startAngle=-45,
#         endAngle=235,
#         color=color.bgr_tuple,
#         thickness=thickness,
#         lineType=cv2.LINE_4
#     )
#     return image


# base annotator


# @dataclass
# class BaseAnnotator:
#     colors: List[Color]
#     thickness: int
#
#     def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
#         annotated_image = image.copy()
#         for detection in detections:
#             annotated_image = draw_ellipse(
#                 image=image,
#                 rect=detection.rect,
#                 color=self.colors[detection.class_id],
#                 thickness=self.thickness
#             )
#         return annotated_image


# text annotator to display tracker_id
# @dataclass
# class TextAnnotator:
#     background_color: Color
#     text_color: Color
#     text_thickness: int
#
#     def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
#         annotated_image = image.copy()
#         for detection in detections:
#             # if tracker_id is not assigned skip annotation
#             if detection.tracker_id is None:
#                 continue
#
#             # calculate text dimensions
#             size, _ = cv2.getTextSize(
#                 str(detection.tracker_id),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 thickness=self.text_thickness)
#             width, height = size
#
#             # calculate text background position
#             center_x, center_y = detection.rect.bottom_center.int_xy_tuple
#             x = center_x - width // 2
#             y = center_y - height // 2 + 10
#
#             # draw background
#             annotated_image = draw_filled_rect(
#                 image=annotated_image,
#                 rect=Rect(x=x, y=y, width=width, height=height).pad(padding=5),
#                 color=self.background_color)
#
#             # draw text
#             annotated_image = draw_text(
#                 image=annotated_image,
#                 anchor=Point(x=x, y=y + height),
#                 text=str(detection.tracker_id),
#                 color=self.text_color,
#                 thickness=self.text_thickness)
#         return annotated_image


# text annotator to display tracker_id
# @dataclass
# class TextAnnotator:
#     background_color: Color
#     text_color: Color
#     text_thickness: int
#
#     def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
#         annotated_image = image.copy()
#         for detection in detections:
#             # if tracker_id is not assigned skip annotation
#             if detection.tracker_id is None:
#                 continue
#
#             # calculate text dimensions
#             size, _ = cv2.getTextSize(
#                 str(detection.tracker_id),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 thickness=self.text_thickness)
#             width, height = size
#
#             # calculate text background position
#             center_x, center_y = detection.rect.bottom_center.int_xy_tuple
#             x = center_x - width // 2
#             y = center_y - height // 2 + 10
#
#             # draw background
#             annotated_image = draw_filled_rect(
#                 image=annotated_image,
#                 rect=Rect(x=x, y=y, width=width, height=height).pad(padding=5),
#                 color=self.background_color)
#
#             # draw text
#             annotated_image = draw_text(
#                 image=annotated_image,
#                 anchor=Point(x=x, y=y + height),
#                 text=str(detection.tracker_id),
#                 color=self.text_color,
#                 thickness=self.text_thickness)
#         return annotated_image
