#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// Structure to hold binding information
struct Binding {
    size_t size;
    size_t dsize;
    nvinfer1::Dims dims;
    std::string name;
};

// External CUDA kernel function (from preprocess.cu)
extern "C" void launch_preprocess_kernel(uint8_t* src, float* dst, int width, int height, cudaStream_t stream);

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

static std::string find_first_existing(const std::vector<std::string>& candidates) {
    for (const auto& p : candidates) {
        if (file_exists(p)) return p;
    }
    return "";
}

class WaferInference {
public:
    WaferInference(const std::string& engine_path) {
        initEngine(engine_path);
        if (!context_) {
            throw std::runtime_error("Failed to initialize TensorRT context");
        }
    }

    ~WaferInference() {
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;
    }

    void infer(const cv::Mat& result_img) {
        // 1. Preprocess (Host -> Device) + Custom Kernel
        // Assume image is already loaded and resized to 224x224
        size_t img_size = 224 * 224 * 3 * sizeof(uint8_t);
        uint8_t* d_img_raw;
        cudaMalloc(reinterpret_cast<void**>(&d_img_raw), img_size);
        cudaMemcpy(d_img_raw, result_img.data, img_size, cudaMemcpyHostToDevice);

        // Call Custom CUDA Kernel for Normalization
        launch_preprocess_kernel(d_img_raw, (float*)d_input_, 224, 224, stream_);

        // 2. Inference
        void* buffers[] = {d_input_, d_output_};
        bool ok = context_->enqueueV2(buffers, stream_, nullptr);
        if (!ok) {
            throw std::runtime_error("TensorRT enqueue failed");
        }

        // 3. Postprocess (Device -> Host)
        std::vector<float> cpu_output(output_size_);
        cudaMemcpyAsync(cpu_output.data(), d_output_, output_byte_size_, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        cudaFree(d_img_raw);

        // Find Max
        auto max_it = std::max_element(cpu_output.begin(), cpu_output.end());
        int class_id = std::distance(cpu_output.begin(), max_it);
        float confidence = *max_it; // Needs softmax if raw logits

        std::cout << "Predicted Class: " << class_id << " (Confidence: " << confidence << ")" << std::endl;
    }

private:
    void initEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Error reading engine file" << std::endl;
            return;
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        char* trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!runtime_) {
            delete[] trtModelStream;
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return;
        }

        engine_ = runtime_->deserializeCudaEngine(trtModelStream, size);
        if (!engine_) {
            delete[] trtModelStream;
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return;
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            delete[] trtModelStream;
            std::cerr << "Failed to create TensorRT execution context" << std::endl;
            return;
        }
        delete[] trtModelStream;

        // Allocate buffers (Assume single input/output for simplicity)
        // Correct logic should iterate bindings
        input_size_ = 1 * 3 * 224 * 224;
        output_size_ = 38;
        
        input_byte_size_ = input_size_ * sizeof(float);
        output_byte_size_ = output_size_ * sizeof(float);

        cudaMalloc(&d_input_, input_byte_size_);
        cudaMalloc(&d_output_, output_byte_size_);
        
        cudaStreamCreate(&stream_);
    }

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    
    size_t input_size_;
    size_t output_size_;
    size_t input_byte_size_;
    size_t output_byte_size_;
    
    cudaStream_t stream_ = nullptr;
};

int main(int argc, char** argv) {
    std::string engine_path;
    if (argc > 1) {
        engine_path = argv[1];
    } else {
        engine_path = find_first_existing({
            "mobilenetv4.engine",
            "../mobilenetv4.engine",
            "../../mobilenetv4.engine",
            "../../../mobilenetv4.engine"
        });
    }

    if (engine_path.empty()) {
        std::cerr << "Cannot find mobilenetv4.engine. Please place it in current/parent directories or pass engine path as argv[1]." << std::endl;
        return 1;
    }

    std::cout << "Using engine: " << engine_path << std::endl;
    WaferInference engine(engine_path);

    cv::Mat img(224, 224, CV_8UC3, cv::Scalar(100, 100, 100));
    std::cout << "Using generated dummy image for benchmark" << std::endl;
    cv::resize(img, img, cv::Size(224, 224));
    
    // Warmup
    for(int i=0; i<10; i++) engine.infer(img);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) {
        engine.infer(img);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Avg Latency: " << elapsed.count() / 100.0 << " ms" << std::endl;

    return 0;
}
