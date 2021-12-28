#from toOnnx import preprocess_image, postprocess
#import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import os
import common



ONNX_FILE_PATH = "./onnx/yolor_p6.onnx"
ENGINE_FILE_PATH = "./engine/yolor_p6_fp16.trt"
# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 28 # 256MiB for 28
            builder.max_batch_size = 1
            builder.fp16_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 896, 896]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():

    # Output shapes expected by the post-processor
    # output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    # # Do inference with TensorRT
    # trt_outputs = []
    # with get_engine(ONNX_FILE_PATH, ENGINE_FILE_PATH) as engine, engine.create_execution_context() as context:
    #     print("getting engine")
    #     inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    #     # Do inference
    #     print('Running inference on image {}...'.format(input_image_path))
    #     # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    #     inputs[0].host = image
    #     trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    # trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    
    # initialize TensorRT engine and parse ONNX model
    engine, context = get_engine(ONNX_FILE_PATH,engine_file_path=ENGINE_FILE_PATH)
    # # get sizes of input and output and allocate memory required for input data and for output data
    # for binding in engine:
    #     if engine.binding_is_input(binding):  # we expect only one input
    #         input_shape = engine.get_binding_shape(binding)
    #         input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
    #         device_input = cuda.mem_alloc(input_size)
    #     else:  # and one output
    #         output_shape = engine.get_binding_shape(binding)
    #         # create page-locked memory buffers (i.e. won't be swapped to disk)
    #         host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
    #         device_output = cuda.mem_alloc(host_output.nbytes)

    # # # Create a stream in which to copy inputs/outputs and run inference.
    # stream = cuda.Stream()


    # # # preprocess input data
    # host_input = np.array(preprocess_image("./turkish_coffee.jpg").numpy(), dtype=np.float32, order='C')
    # cuda.memcpy_htod_async(device_input, host_input, stream)

    # # # run inference
    # context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    # cuda.memcpy_dtoh_async(host_output, device_output, stream)
    # stream.synchronize()

    # # # postprocess results
    # output_data = torch.Tensor(host_output).reshape(engine.max_batch_size, output_shape[0])
    # postprocess(output_data)


if __name__ == '__main__':
    main()
