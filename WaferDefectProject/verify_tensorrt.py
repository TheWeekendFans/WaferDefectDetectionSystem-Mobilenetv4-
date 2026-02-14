import argparse
import os
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, use_fp16=True):
    print(f"Building TensorRT Engine from {onnx_file_path}...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Check if FP16 is supported
    if use_fp16 and builder.platform_has_fast_fp16:
        print("FP16 support detected. Enabling FP16.")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
            
    # Build Engine
    # Note: set_memory_pool_limit is for TensorRT 8.5+
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
    except:
        pass # Older versions use max_workspace_size
        
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
        
    print(f"Engine saved to {engine_file_path}")
    return serialized_engine

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            size = trt.volume(shape) * np.dtype(dtype).itemsize

            d_ptr = cuda.mem_alloc(size)
            self.bindings.append(int(d_ptr))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': tensor_name, 'host': None, 'device': d_ptr, 'shape': shape, 'dtype': dtype})
            else:
                self.outputs.append({'name': tensor_name, 'host': cuda.pagelocked_empty(trt.volume(shape), dtype), 'device': d_ptr, 'shape': shape})
                
    def infer(self, image_np):
        cuda.memcpy_htod_async(self.inputs[0]['device'], image_np, self.stream)

        self.context.set_tensor_address(self.inputs[0]['name'], self.bindings[0])
        self.context.set_tensor_address(self.outputs[0]['name'], self.bindings[1])

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()

        return self.outputs[0]['host']

def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT build and latency check")
    parser.add_argument("--onnx", type=str, default="mobilenetv4_sim.onnx")
    parser.add_argument("--onnx-fallback", type=str, default="mobilenetv4.onnx")
    parser.add_argument("--engine", type=str, default="mobilenetv4.engine")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    onnx_path = args.onnx if os.path.exists(args.onnx) else args.onnx_fallback
    engine_path = args.engine

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if not os.path.exists(engine_path):
        build_engine(onnx_path, engine_path, use_fp16=args.fp16)

    trt_infer = TensorRTInference(engine_path)

    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for _ in range(args.warmup):
        trt_infer.infer(dummy_input)

    start = time.time()
    for _ in range(args.runs):
        trt_infer.infer(dummy_input)
    end = time.time()

    print(f"TensorRT Inference Latency: {(end - start) / args.runs * 1000:.2f} ms")
