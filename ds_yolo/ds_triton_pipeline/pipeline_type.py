import gi  
import os
import pathlib
import sys

file_path = pathlib.Path(__file__).resolve().parents[1]

sys.path.append(os.path.abspath(file_path))

gi.require_version("Gst", "1.0")
from gi.repository import GObject, GLib, Gst
from loguru import logger

config_path = os.path.join(file_path, "config_ds_triton_infer")
ds_dali_yolo_config = os.path.join(config_path, "ds_dali_yolov5_trt_nopostprocess.txt")
ds_yolo_config = os.path.join(config_path, "ds_yolov5_trt_nopostprocess.txt")
grpc_ds_dali_yolo_config = os.path.join(config_path, "grpc_ds_dali_yolov5_trt_nopostprocess.txt")
grpc_ds_yolo_config = os.path.join(config_path, "grpc_ds_yolov5_trt_nopostprocess.txt")


# PIPELINE FOR H264 DEEPSTREAM TRITON INFERENCE

def h264_pipeline(
    pipeline, pl, 
    test_video_file,
    batch_size=1,  
    is_save_output=True, 
    output_video_name="./out.mp4", 
    image_width=1920, image_height=1080, 
    is_dali=False, is_grpc=False):
    """
    Build pipeline for inference h264 stream video input
    """
    # Source element for reading from the file
    source = pl.make_elm_or_print_err("filesrc", "file-source", "Source")

    # Since the data format in the input file is elementary h264 stream,
    # we need a h264parser
    h264parser = pl.make_elm_or_print_err("h264parse", "h264-parser", "H264Parser")
    parser = h264parser

    # Use nvdec_h264 for hardware accelerated decode on GPU
    decoder = pl.make_elm_or_print_err("nvv4l2decoder", "nvv4l2-decoder", "Decoder")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = pl.make_elm_or_print_err("nvstreammux", "Stream-muxer", "NvStreamMux")

    # Use nvinferserver to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = pl.make_elm_or_print_err("nvinferserver", "primary-inference", "Nvinferserver")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = pl.make_elm_or_print_err("nvvideoconvert", "convertor", "Nvvidconv")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = pl.make_elm_or_print_err("nvdsosd", "onscreendisplay", "OSD (nvosd)")

    # Finally encode and save the osd output
    queue = pl.make_elm_or_print_err("queue", "queue", "Queue")

    nvvidconv2 = pl.make_elm_or_print_err("nvvideoconvert", "convertor2", "Converter 2 (nvvidconv2)")

    capsfilter = pl.make_elm_or_print_err("capsfilter", "capsfilter", "capsfilter")

    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    # On Jetson, there is a problem with the encoder failing to initialize
    # due to limitation on TLS usage. To work around this, preload libgomp.
    # Add a reminder here in case the user forgets.
    preload_reminder = "If the following error is encountered:\n" + \
                       "/usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block\n" + \
                       "Preload the offending library:\n" + \
                       "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1\n"
    encoder = pl.make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder", preload_reminder)

    encoder.set_property("bitrate", 2000000)

    codeparser = pl.make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')

    container = pl.make_elm_or_print_err("qtmux", "qtmux", "Container")

    if is_save_output:
        sink = pl.make_elm_or_print_err("filesink", "filesink", "Sink")

        sink.set_property("location", output_video_name)
        sink.set_property("sync", 0)
        sink.set_property("async", 0)
    else:
        sink = pl.make_elm_or_print_err("fakesink", "fake-sink", "FakeSink")

    logger.info("Playing file %s " % test_video_file)
    source.set_property("location", test_video_file)
    streammux.set_property("width", image_width)
    streammux.set_property("height", image_height)
    streammux.set_property("batch-size", batch_size)
    streammux.set_property("batched-push-timeout", 4000000)
    if not is_dali:
        if not is_grpc:
            logger.info("DeepStream Triton yolov5 tensorRT inference")
            pgie.set_property("config-file-path", ds_yolo_config)
        else:
            logger.info("DeepStream GRPC Triton yolov5 tensorRT inference")
            pgie.set_property("config-file-path", grpc_ds_yolo_config)
    else:
        if not is_grpc:
            logger.info("DeepStream Triton DALI yolov5 tensorRT inference")
            pgie.set_property("config-file-path", ds_dali_yolo_config)
        else:

            logger.info("DeepStream GRPC Triton DALI yolov5 tensorRT inference")
            pgie.set_property("config-file-path", grpc_ds_dali_yolo_config)

    logger.info("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    if is_save_output:
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(queue)
        pipeline.add(nvvidconv2)
        pipeline.add(capsfilter)
        pipeline.add(encoder)
        pipeline.add(codeparser)
        pipeline.add(container)
    pipeline.add(sink)

    # we link the elements together
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    logger.info("Linking elements in the Pipeline \n")
    source.link(parser)
    parser.link(decoder)

    # streammux link to decoded frame sink_0
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        logger.opt(colors=True).warning(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        logger.opt(colors=True).warning(" Unable to get source pad of decoder \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)

    if is_save_output:
        pgie.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(queue)
        queue.link(nvvidconv2)
        nvvidconv2.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(codeparser)
        codeparser.link(container)
        container.link(sink)
    else:
        pgie.link(sink)
    
    return pipeline, pgie, nvosd


#----------------------------------------------------------------------------------
# PIPELINE FOR URI OR LOCAL VIDEO FILE

def cb_newpad(decodebin, decoder_src_pad, data):
    logger.info("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    logger.info("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        logger.info("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                logger.opt(colors=True).warning("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            logger.opt(colors=True).warning(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    logger.info("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(uri):
    """
    Decode file video local and URI video input 
    """
    logger.info("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-0"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.opt(colors=True).warning(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        logger.opt(colors=True).warning(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.opt(colors=True).warning(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

def uri_local_pipeline(
    pipeline, pl, 
    test_video_file, 
    batch_size=1, 
    is_save_output=True, 
    output_video_name="./out.mp4", 
    image_width=1920, image_height=1080, 
    is_dali=False,
    is_grpc=False):
    """
    Build Pipeline for inference mp4 and URI video 
    """
    # Source element for reading from the uri or local file
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = pl.make_elm_or_print_err("nvstreammux", "Stream-muxer", "NvStreamMux")

    # Use nvinferserver to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = pl.make_elm_or_print_err("nvinferserver", "primary-inference", "Nvinferserver")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = pl.make_elm_or_print_err("nvvideoconvert", "convertor", "Nvvidconv")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = pl.make_elm_or_print_err("nvdsosd", "onscreendisplay", "OSD (nvosd)")


    # Finally encode and save the osd output
    queue = pl.make_elm_or_print_err("queue", "queue", "Queue")

    nvvidconv2 = pl.make_elm_or_print_err("nvvideoconvert", "convertor2", "Converter 2 (nvvidconv2)")

    capsfilter = pl.make_elm_or_print_err("capsfilter", "capsfilter", "capsfilter")

    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    # On Jetson, there is a problem with the encoder failing to initialize
    # due to limitation on TLS usage. To work around this, preload libgomp.
    # Add a reminder here in case the user forgets.
    preload_reminder = "If the following error is encountered:\n" + \
                       "/usr/lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block\n" + \
                       "Preload the offending library:\n" + \
                       "export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1\n"
    encoder = pl.make_elm_or_print_err("avenc_mpeg4", "encoder", "Encoder", preload_reminder)

    encoder.set_property("bitrate", 2000000)

    codeparser = pl.make_elm_or_print_err("mpeg4videoparse", "mpeg4-parser", 'Code Parser')

    container = pl.make_elm_or_print_err("qtmux", "qtmux", "Container")

    if is_save_output:
        sink = pl.make_elm_or_print_err("filesink", "filesink", "Sink")

        sink.set_property("location", output_video_name)
        sink.set_property("sync", 0)
        sink.set_property("async", 0)
    else:
        sink = pl.make_elm_or_print_err("fakesink", "fake-sink", "FakeSink")

    logger.info("Playing URI file %s " % test_video_file)
    uri_name = test_video_file
    source_bin = create_source_bin(uri_name)
    if not source_bin:
        logger.opt(colors=True).warning("Unable to create source bin \n")
    padname = "sink_0"
    
    
    streammux.set_property("width", image_width)
    streammux.set_property("height", image_height)
    streammux.set_property("batch-size", batch_size)
    streammux.set_property("batched-push-timeout", 4000000)
    if not is_dali:
        if not is_grpc:
            logger.info("DeepStream Triton yolov5 tensorRT inference")
            pgie.set_property("config-file-path", ds_yolo_config)
        else:
            logger.info("DeepStream GRPC Triton yolov5 tensorRT inference")
            pgie.set_property("config-file-path", grpc_ds_yolo_config)
    else:
        if not is_grpc:
            logger.info("DeepStream Triton DALI yolov5 tensorRT inference")
            pgie.set_property("config-file-path", ds_dali_yolo_config)
        else:

            logger.info("DeepStream GRPC Triton DALI yolov5 tensorRT inference")
            pgie.set_property("config-file-path", grpc_ds_dali_yolo_config)

    logger.info("Adding elements to Pipeline \n")
    pipeline.add(source_bin)
    pipeline.add(streammux)
    pipeline.add(pgie)

    if is_save_output:
        pipeline.add(nvvidconv)
        pipeline.add(nvosd)
        pipeline.add(queue)
        pipeline.add(nvvidconv2)
        pipeline.add(capsfilter)
        pipeline.add(encoder)
        pipeline.add(codeparser)
        pipeline.add(container)
    pipeline.add(sink)

    # we link the elements together
    # uri -> uridecode bin ->
    # nvinfer -> nvvidconv -> nvosd -> video-renderer
    logger.info("Linking elements in the Pipeline \n")

    # streamux link to decode frame by padname
    sinkpad = streammux.get_request_pad(padname)
    if not sinkpad:
        logger.warning("Unable to create sink pad bin \n")
    
    srcpad = source_bin.get_static_pad("src")
    if not srcpad:
        logger.warning("Unable to create src pad bin \n")
    
    srcpad.link(sinkpad)
    streammux.link(pgie)
    if is_save_output:
        pgie.link(nvvidconv)
        nvvidconv.link(nvosd)
        nvosd.link(queue)
        queue.link(nvvidconv2)
        nvvidconv2.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(codeparser)
        codeparser.link(container)
        container.link(sink)
    else:
        pgie.link(sink)
    
    return pipeline, pgie, nvosd