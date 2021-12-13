import argparse
import pyds
import sys

sys.path.append("../")

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst
from loguru import logger

from common.bus_call import bus_call

from pipeline_parts import PipleParts
from pipeline_type import h264_pipeline, uri_local_pipeline


parser = argparse.ArgumentParser(description="Deepstream Triton Yolov5 PIPELINE")
parser.add_argument(
    "--test_video", help="test video file path or uri", type=str, 
    default="/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_qHD.h264")
parser.add_argument("--label_type", help="Label type (coco)", type=str, default="coco")
parser.add_argument("--is_save", help="Save result video output", action="store_true")
parser.add_argument("--conf", help="Confidence threshold for YOLOv5", type=float, default=0.5)
parser.add_argument("--iou", help="IOU threshold", type=float, default=0.45)
parser.add_argument("--outvid_width", help="VIDEO OUTPUT WIDTH", type=int, default=1920)
parser.add_argument("--outvid_height", help="VIDEO OUTPUT_HEIGHT", type=int, default=1080)
parser.add_argument("--is_dali", help="DeepStream Triton DALI yolov5 trt, use dali for preprocessing", action="store_true")

args = parser.parse_args()

IS_SAVE = args.is_save
IMAGE_HEIGHT = args.outvid_height
IMAGE_WIDTH = args.outvid_width
OUTPUT_VIDEO_NAME = "./ds_triton_yolov5_trt_out.mp4"
CONF_THRESHOLD = args.conf
IOU_THRESHOLD = args.iou
IS_DALI = args.is_dali
test_video = args.test_video
label_type = args.label_type


def ds_pipeline(
    test_video, 
    conf_threshold=0.5, iou_threshold=0.45, 
    is_save_output=False, 
    outvid_width=1920, outvid_height=1080, 
    is_dali=False, 
    output_video_name="./ds_triton_yolov5_trt_out.mp4", 
    label_type="flag"):
    # Initialize Pipleline parts
    pl = PipleParts(
        is_save_output=IS_SAVE, 
        image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, 
        conf_threshold=CONF_THRESHOLD, nms_threshold=IOU_THRESHOLD, 
        label_type=label_type)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    try:
        if test_video.endswith(".h264") and not "https://" in test_video:
            pipeline, pgie, nvosd = h264_pipeline(
                pipeline, pl, 
                test_video, 
                is_save_output=IS_SAVE, 
                output_video_name=OUTPUT_VIDEO_NAME, 
                image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                is_dali=IS_DALI)
        elif "file://" in test_video or "https://" in test_video:
            pipeline, pgie, nvosd = uri_local_pipeline(
                pipeline, pl, 
                test_video, 
                is_save_output=IS_SAVE, 
                output_video_name=OUTPUT_VIDEO_NAME, 
                image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, 
                is_dali=IS_DALI)
    except Exception as ex:
        logger.error(ex)
        sys.exit(1)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Add a probe on the primary-infer source pad to get inference output tensors
    pgiesrcpad = pgie.get_static_pad("src")
    if not pgiesrcpad:
        sys.stderr.write(" Unable to get src pad of primary infer \n")

    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pl.pgie_src_pad_buffer_probe, 0)

    if IS_SAVE:
        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        osdsinkpad = nvosd.get_static_pad("sink")
        if not osdsinkpad:
            sys.stderr.write(" Unable to get sink pad of nvosd \n")

        osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, pl.osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    sys.exit(
        ds_pipeline(
            test_video=test_video, 
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            is_save_output=IS_SAVE,
            outvid_width=IMAGE_WIDTH,
            outvid_height=IMAGE_HEIGHT,
            is_dali=IS_DALI,
            output_video_name=OUTPUT_VIDEO_NAME,
            label_type=label_type
        ))
