import argparse
import os
import pyds
import time
import sys

sys.path.append("../")

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst
from loguru import logger

from common.bus_call import bus_call

from ds_triton_pipeline.pipeline_parts import PipelineParts
from ds_triton_pipeline.pipeline_type import uri_local_pipeline


parser = argparse.ArgumentParser(description="Deepstream Triton Yolov5 PIPELINE")
parser.add_argument(
    "--test_video", 
    help="""
    test video file path or uri or jpg image. 
    The system allow a or more source and accepts any format like: mp4, h264, https, rstp... 
    Setup `--test_video` file:///path/to/video_1.mp4 file:///path/to/video_2.mp4""", 
    type=str, nargs="+",
    default=["/opt/nvidia/deepstream/deepstream-6.0/samples/streams/sample_qHD.h264"])
parser.add_argument("--batch_size", help="batch size inference", type=int, default=1)
parser.add_argument("--label_type", help="Label type (flag/nsfw)", type=str, default="flag")
parser.add_argument("--is_save", help="Save result video output", action="store_true")
parser.add_argument("--conf", help="Confidence threshold for YOLOv5", type=float, default=0.5)
parser.add_argument("--iou", help="IOU threshold", type=float, default=0.45)
parser.add_argument("--outvid_width", help="VIDEO OUTPUT WIDTH", type=int, default=1920)
parser.add_argument("--outvid_height", help="VIDEO OUTPUT_HEIGHT", type=int, default=1080)
parser.add_argument("--is_dali", help="DeepStream Triton DALI yolov5 trt, use dali for preprocessing", action="store_true")
parser.add_argument(
    "--is_grpc", help="Don't use DeepStream Triton Docker, use seperately DeepStream and Triton that need GRPC protocol for communicate", action="store_true")

args = parser.parse_args()

IS_SAVE = args.is_save
IMAGE_HEIGHT = args.outvid_height
IMAGE_WIDTH = args.outvid_width
OUTPUT_VIDEO_NAME = "./ds_triton_yolov5_trt_out.mp4"
CONF_THRESHOLD = args.conf
IOU_THRESHOLD = args.iou
IS_DALI = args.is_dali
IS_GRPC = args.is_grpc
test_video = args.test_video
label_type = args.label_type
BATCH_SIZE = args.batch_size


def ds_pipeline(
    test_video, 
    batch_size=1,
    conf_threshold=0.5, iou_threshold=0.45, 
    is_save_output=False, 
    outvid_width=1920, outvid_height=1080, 
    is_dali=False, 
    is_grpc=False,
    output_video_name="./ds_triton_yolov5_trt_out.mp4", 
    label_type="flag"):
    """
    test_video: path to test file (.mp4, .h264, https://, rtsp://, .jpg, mjpeg)
    batch_size: batch size
    conf_threshold: confidence threshold for postprocess
    iou_threshold: iou threshold for nms
    is_save_output: bool for save video output
    outvid_width, outvid_height: output video width, height
    output_video_name: path to output video 
    label_type: type of label (flag/nsfw)  
    """
    init_t = time.perf_counter()
    # can't save video output if batch size larger than 1
    if batch_size != 1 or len(test_video) != 1:
        if is_save_output:
            logger.warning("Save video output only operate with 1 video input and batch_size == 1")
            is_save_output = False

    # Initialize Pipleline parts
    try:
        pl = PipelineParts(
            is_save_output=is_save_output, 
            image_width=outvid_width, image_height=outvid_height, 
            conf_threshold=conf_threshold, nms_threshold=iou_threshold, 
            label_type=label_type)
    except Exception as ex:
        logger.error(f"ERROR: {ex}")

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    logger.info("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        logger.warning("WARNING: Unable to create Pipeline \n")
    try:
        # uridecoders -> streammux -> triton_infer -> postprocess 
        pipeline, pgie, nvosd = uri_local_pipeline(
            pipeline, pl, 
            test_video, 
            batch_size=batch_size,
            is_save_output=is_save_output, 
            output_video_name=output_video_name, 
            image_width=outvid_width, image_height=outvid_height, 
            is_dali=is_dali, is_grpc=is_grpc)

    except Exception as ex:
        logger.error(f"ERROR: {ex}")
        logger.debug(
            "DEBUG: The system allow a or more source and accepts any format like: mp4, h264, https, rstp... \nSetup --test_video file:///path/to/video_1.mp4 file:///path/to/video_2.mp4 ")
        sys.exit(1)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Add a probe on the primary-infer source pad to get inference output tensors
    try:
        pgiesrcpad = pgie.get_static_pad("src")
        if not pgiesrcpad:
                logger.warning("WARNING: Unable to get src pad of primary infer \n")
    except Exception as ex:
        logger.error("ERROR: {}".format(ex))
        sys.exit(1)
    

    pgiesrcpad.add_probe(Gst.PadProbeType.BUFFER, pl.pgie_src_pad_buffer_probe, 0)

    if is_save_output:
        # Lets add probe to get informed of the meta data generated, we add probe to
        # the sink pad of the osd element, since by that time, the buffer would have
        # had got all the metadata.
        try:
            osdsinkpad = nvosd.get_static_pad("sink")
            if not osdsinkpad:
                logger.warning("WARNING: Unable to get sink pad of nvosd \n")
        
            osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, pl.osd_sink_pad_buffer_probe, 0)    
        except Exception as ex:
            logger.error("ERROR: {}".format(ex))    
    
    done_init_t = time.perf_counter() - init_t

    # start play back and listen to events
    logger.info("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    
    start_t = time.perf_counter()
    try:
        loop.run()
    except Exception as ex:
        logger.warning(f"WARNING: {ex}")
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)
    
    end = time.perf_counter() - start_t
    
    logger.info(f"[INFO] Initialize pipeline time: {done_init_t:.5f}")
    logger.info(f"[INFO] Inference + Postprocess + save (option): {end:.5f}")
    logger.info(f"[INFO] elapsed time: {(done_init_t + end):.5f}")
    logger.info(f"[INFO] extracted frames: {pl.extracted_frame}")


if __name__ == "__main__":
    """
    Process at client:
    * Takes a path to a file media or uri.

    * Gstreamer initialization is performed.

    * Several  elements a created in order to make a pipeline.

    * These elements are added to the pipeline and linked together.

    * Probe functions are linked to the pipeline in order to interact with the data:

        * **pgie_src_pad_buffer_probe**: get tensor metadata from triton inference output, convert to numpy array and yolo postprocessing.

        * **osd_sink_pad_buffer_probe**: encode frame and save to video output.

    * The pipeline is set to its PLAYING mode.

    * The main loop is run.

    * The pipeline is set to its NULL mode.
    """
    sys.exit(
        ds_pipeline(
            test_video=test_video, 
            batch_size=BATCH_SIZE,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            is_save_output=IS_SAVE,
            outvid_width=IMAGE_WIDTH,
            outvid_height=IMAGE_HEIGHT,
            is_dali=IS_DALI,
            is_grpc=IS_GRPC,
            output_video_name=OUTPUT_VIDEO_NAME,
            label_type=label_type
        ))
