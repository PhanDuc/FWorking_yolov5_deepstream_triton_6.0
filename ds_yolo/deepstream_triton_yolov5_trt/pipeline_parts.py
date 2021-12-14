import ctypes
import gi
import numpy as np
import io 
import pyds
import sys

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

gi.require_version("Gst", "1.0")
from gi.repository import GObject, GLib, Gst
from loguru import logger

from labels import NSFWLabels, FlagLabels
from trt_postprocess import postprocess


class PipelineParts():
    def __init__(
        self,   
        conf_threshold=0.5, 
        nms_threshold=0.4, 
        is_save_output=True, 
        image_width=1920, image_height=1080, 
        label_type="flag"):
        
        self.conf_thresh = conf_threshold
        self.nms_thresh = nms_threshold
        self.is_save_output=is_save_output
        self.image_width = image_width
        self.image_height = image_height
        
        if label_type == "flag":
            self.label = FlagLabels
        elif label_type == "nsfw":
            self.label = NSFWLabels
        
    def make_elm_or_print_err(self, factoryname, name, printedname, detail=""):
        """ Creates an element with Gst Element Factory make.
            Return the element  if successfully created, otherwise print
            to stderr and return None.
        """
        print("Creating", printedname)
        elm = Gst.ElementFactory.make(factoryname, name)
        if not elm:
            sys.stderr.write("Unable to create " + printedname + " \n")
            if detail:
                sys.stderr.write(detail)
        return elm


    def osd_sink_pad_buffer_probe(self, pad, info, u_data):
        """
        Encode frame and save to video output. 
        """
        
        frame_number = 0
        # Intiallizing object counter with 0.
        num_rects = 0

        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            frame_number = frame_meta.frame_num
            num_rects = frame_meta.num_obj_meta
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    # Casting l_obj.data to pyds.NvDsObjectMeta
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            # Acquiring a display meta object. The memory ownership remains in
            # the C code so downstream plugins can still access it. Otherwise
            # the garbage collector will claim it when this probe function exits.
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            py_nvosd_text_params = display_meta.text_params[0]
            # Setting display text to be shown on screen
            # Note that the pyds module allocates a buffer for the string, and the
            # memory will not be claimed by the garbage collector.
            # Reading the display_text field here will return the C address of the
            # allocated string. Use pyds.get_string() to get the string content.

            disp_string = "Frame Number={} Number of Objects={}"
            
            py_nvosd_text_params.display_text = disp_string.format(
                frame_number,
                num_rects,
            )

            # Now set the offsets where the string should appear
            py_nvosd_text_params.x_offset = 10
            py_nvosd_text_params.y_offset = 12

            # Font , font-color and font-size
            py_nvosd_text_params.font_params.font_name = "Serif"
            py_nvosd_text_params.font_params.font_size = 10
            # set(red, green, blue, alpha); set to White
            py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

            # Text background color
            py_nvosd_text_params.set_bg_clr = 1
            # set(red, green, blue, alpha); set to Black
            py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
            # Using pyds.get_string() to get display_text as string
            print(pyds.get_string(py_nvosd_text_params.display_text))
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK


    def add_obj_meta_to_frame(self, bbox, score, label, batch_meta, frame_meta):
        """ 
        Inserts an object into the metadata 
        """
        untracked_obj_id=0xffffffffffffffff
        # this is a good place to insert objects into the metadata.
        # Here's an example of inserting a single object.
        obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
        # Set bbox properties. These are in input resolution.
        rect_params = obj_meta.rect_params
        rect_params.left = int(bbox[0])
        rect_params.top = int(bbox[1])
        rect_params.width = int((bbox[2] - bbox[0]))
        rect_params.height = int(bbox[3] - bbox[1])
        #logger.info(f"rect_params.left: {rect_params.height}")

        # Semi-transparent yellow backgroud
        rect_params.has_bg_color = 0
        rect_params.bg_color.set(1, 1, 0, 0.4)

        # Red border of width 3
        rect_params.border_width = 3
        rect_params.border_color.set(1, 0, 0, 1)

        # Set object info including class, detection confidence, etc.
        obj_meta.confidence = score
        #obj_meta.class_id = int(label)

        # There is no tracking ID upon detection. The tracker will
        # assign an ID.
        obj_meta.object_id = untracked_obj_id

        #lbl_id = int(label)
        #if lbl_id >= len(label_names):
        #    lbl_id = 0

        # Set the object classification label.
        obj_meta.obj_label = label

        # Set display text for the object.
        txt_params = obj_meta.text_params
        if txt_params.display_text:
            pyds.free_buffer(txt_params.display_text)

        txt_params.x_offset = int(rect_params.left)
        txt_params.y_offset = max(0, int(rect_params.top) - 10)
        txt_params.display_text = (
            label + " " + "{:04.3f}".format(score)
        )
        # Font , font-color and font-size
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        txt_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        # Inser the object into current frame meta
        # This object has no parent
        pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)


    def pgie_src_pad_buffer_probe(self, pad, info, u_data):
        """
        Get tensor_metadata from triton inference output 
        Convert tensor metadata to numpy array 
        Postprocessing 
        """
        # get the buffer of info argument
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return
        logger.info(f"gst_buffer: {gst_buffer}")
        logger.info(f"hash gst_buffer: {hash(gst_buffer)}")

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        
        #logger.info("batch_meta: {}".format(batch_meta))
        
        l_frame = batch_meta.frame_meta_list

        logger.info(f"l_frame.num_frames_in_batch: {batch_meta.num_frames_in_batch}")

        #label_names = self.get_label_names_from_file()

        
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                #logger.info("frame_meta: {}".format(frame_meta))
            except StopIteration:
                break
            
            l_user = frame_meta.frame_user_meta_list
            
            if not self.is_save_output:
                # get width and height of source video 
                src_width = frame_meta.source_frame_width
                src_height = frame_meta.source_frame_height
                #logger.info(f"source_height: {src_height}")
                #logger.info(f"source_width: {src_width}")

                height_prp = src_height
                width_prp = src_width
            else: 
                height_prp = self.image_height
                width_prp = self.image_width 

            
            
            while l_user is not None:
                try:
                    # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                    # The casting also keeps ownership of the underlying memory
                    # in the C code, so the Python garbage collector will leave
                    # it alone.
                    user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                except StopIteration:
                    break

                if (
                        user_meta.base_meta.meta_type
                        != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META
                ):
                    #logger.info("user_meta: {}".format(user_meta))
                    continue
                
                # get tensor-meta from triton inference output
                tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                #logger.info("tensor_meta: {}".format(tensor_meta))

                # Boxes in the tensor meta should be in network resolution which is
                # found in tensor_meta.network_info. Use this info to scale boxes to
                # the input frame resolution.
                layers_info = []
                #logger.info(f"tensor_meta: {tensor_meta.num_output_layers}")
                for i in range(tensor_meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
                    
                    # Convert tensor metadata to numpy array
                    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
                    output_np_array = np.ctypeslib.as_array(ptr, shape=(1, 6001, 1, 1))
                    #logger.info(f"--> np_array.shape: {output_np_array.shape}")
                    layers_info.append(layer)
                #logger.info(f"layers_info: {layers_info}")
                
                detected_obj = postprocess(
                    output_np_array, 
                    width_prp, height_prp,
                    conf_threshold=self.conf_thresh, 
                    nms_threshold=self.nms_thresh)

                bboxes = [[int(box.x1), int(box.y1), int(box.x2), int(box.y2)] for box in detected_obj]
                labels = [self.label(int(box.classID)).name for box in detected_obj]
                scores = [box.confidence for box in detected_obj]

                ensemble_results = {
                    "frame_%s"%frame_meta.frame_num:{
                        "scores":scores, 
                        "labels":labels, 
                        "boxes":bboxes}
                }
                logger.info(f"--> ensemble_result: {ensemble_results}")

                try:
                    l_user = l_user.next
                except StopIteration:
                    break

                # If save video output is true, add bbox and score, label information to frame                     
                if self.is_save_output:
                    for bbox, score, label in zip(bboxes, scores, labels): 
                        #logger.info(f"bounding box: {bbox}")
                        self.add_obj_meta_to_frame(bbox, score, label, batch_meta, frame_meta)

            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK
