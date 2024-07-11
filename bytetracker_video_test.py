from ultralytics import YOLO
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator

import numpy as np
import cv2
from tqdm.notebook import tqdm
from argparse import ArgumentParser

from utils import detections2boxes, match_detections_with_tracks
from yolox.tracker.byte_tracker import BYTETracker

parser = ArgumentParser(description="Test a Bytetracker on the sample video")
parser.add_argument("--source_video_path", type=str, required=True, help="sample video for the test")
parser.add_argument("--model", type=str, default="weights/yolov8x.pt", help="weight file")
parser.add_argument("--output_video_path", type=str, required=True, help="output video file path")
# parser.add_argument("--class_id")

CLASS_ID = [2, 3, 5, 7]
# CLASS_ID = [0]

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False



def main(args):
    model = YOLO(args.model)
    model.fuse()

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    # class_ids of interest - car, motorcycle, bus and truck\

    LINE_START = Point(50, 1500)
    LINE_END = Point(3840-50, 1500)

    # create frame generator
    generator = get_video_frames_generator(args.source_video_path)
    # create instance of BoxAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    # acquire first video frame
    iterator = iter(generator)
    frame = next(iterator)
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame)
    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # format custom labels
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # annotate and display frame
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)



    # create BYTETracker instance
    byte_tracker = BYTETracker(BYTETrackerArgs())
    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(args.source_video_path)
    # create frame generator
    generator = get_video_frames_generator(args.source_video_path)
    # create LineCounter instance
    line_counter = LineCounter(start=LINE_START, end=LINE_END)
    # create instance of BoxAnnotator and LineCounterAnnotator
    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=4, text_thickness=4, text_scale=2)
    line_annotator = LineCounterAnnotator(thickness=4, text_thickness=4, text_scale=2)

    # open target video file
    with VideoSink(args.output_video_path, video_info) as sink:
        # loop over video frames
        for frame in tqdm(generator, total=video_info.total_frames):
            # model prediction on single frame and conversion to supervision Detections
            results = model(frame)
            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            # filtering out detections with unwanted classes
            mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)
            # filtering out detections without trackers
            mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)
            # format custom labels
            labels = [
                f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # updating line counter
            line_counter.update(detections=detections)
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
            line_annotator.annotate(frame=frame, line_counter=line_counter)

            # Display the frame
            resized_frame = cv2.resize(frame, (1280, 720))  # 원하는 크기로 조정 (예: 1280x720)
            cv2.imshow('Frame', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sink.write_frame(frame)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
