This guide shows how to use custom deep learning models and parse their inference output in a Python application. Custom model support is provided by the Triton Inference Server plugin included in the DeepStream SDK. The raw inference output can be parsed in a Python application via access to the NvDsInferTensorMeta structure. Sample code is provided in the deepstream-ssd-parser Python application.

**1 - Inference via Triton**

To perform inference using Triton and access the raw output, the general steps are:

1. Include Triton plugin in the pipeline. The deepstream-ssd-parser app shows how to construct a pipeline with Triton as the primary detector.

2. Configure the Triton plugin to attach raw output tensors to the metadata, and allow for custom post-processing. See details below.

3. Add a probe function to intercept output tensors for parsing. See the deepstream-ssd-parser app for example.

<b>Configuring Triton for custom parsing</b>:

1. Select custom post-processing:

   In ds_test_ssd_no_postprocess.txt, edit the `infer_config -> postprocess` section 
   to specify "other" as the parsing function and point to the custom labels file used for parsing:

   ```

   postprocess {

      Labelfile_path: "../../../../samples/trtis_model_repo/ssd_inception_v2_coco_2018_01_28/labels.txt"

      other {}

    }

   ```

2. Enable attachment of inference output tensors to metadata:

   Add the following text to the config file:

   ```

   output_control {

      output_tensor_meta: true

   }

   ```

**2 - How to intercept tensor meta in a Python application**

This is done in `pgie_src_pad_buffer_probe`.

This is done in several steps:

1. Get the buffer of info argument. 

   ```gst_buffer = info.get_buffer()```

2. Retrieve the gst buffer from batch meta

   ```batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))```

3. Retrieve the first frame

   ```l_frame = batch_meta.frame_meta_list```

4. Retrieve the metadata for the first frame

   ```frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)```

5. Retrieve the frame meta user list and verify it is not None

   ```l_user = frame_meta.frame_user_meta_list```

6. Retrieve the user metadata

   ```user_meta = pyds.NvDsUserMeta.cast(l_user.data)```

7. Retrieve the tensor meta

   ```tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)```

The function ```pgie_src_pad_buffer_probe``` is given as a probe in the pipeline created in main function.

Sample code is available for SSD parser neural network. 

**3 - Tensor meta structure and bindings API**

* pyds.NvDsInferTensorMeta.cast(data): This function casts the data into the pyds.NvDsInferTensorMeta object, this metadata is added as NvDsUserMeta to the frame_user_meta_list of the corresponding frame_meta or object_user_meta_list of the corresponding object with the meta_type set to NVDSINFER_TENSOR_OUTPUT_META.

This object has the following methods and members:

* gpu_id: GPU device ID on which the device buffers have been allocated.

* num_output_layers: Number of output layers.

* out_buf_ptrs_dev: Array of objects to the output device buffers for the frame / object.

* out_buf_ptrs_host: Array of objects to the output host buffers for the frame / object.

* output_layers_info:

				arg0: object of type pyds.NvDsInferTensorMeta

				arg1: integer ‘i’

				Output: returns ith layer in the array.

* priv_data: Private data used for the meta producer’s internal memory management.

* unique_id: Unique ID of the gst-nvinfer instance which attached this meta.

* pyds.NvDsInferObjectDetectionInfo: Holds information about one parsed object from detector’s output.

  This object has the following methods and members:

* classId: ID of the class to which the object belongs.

* detectionConfidence: Object detection confidence. Should be a float value in the range [0,1].

* height: Height of the bounding box shape for the object.

* left: Horizontal offset of the bounding box shape for the object.

* top: Vertical offset of the bounding box shape for the object.

* width: Width of the bounding box shape for the object.

**  **


Below is a general explanation of the deepstream-ssd-parser sample application. See comments in the code for additional details.

* This function takes a path to a file media or uri.

* Gstreamer initialization is performed.

* Several  elements a created in order to make a pipeline.

* These elements are added to the pipeline and linked together.

* Probe functions are linked to the pipeline in order to interact with the data:

    * **pgie_src_pad_buffer_probe**: get tensor metadata from triton inference output, convert to numpy array and yolo postprocessing.

    * **osd_sink_pad_buffer_probe**: encode frame and save to video output.

* The pipeline is set to its PLAYING mode.

* The main loop is run.

* The pipeline is set to its NULL mode.

