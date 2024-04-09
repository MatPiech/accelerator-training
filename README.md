# Accelerated Continual Learning

Requirements:
- `onnxruntime==1.16.3` build with below command

References:
- [ONNX opset support](https://onnxruntime.ai/docs/reference/compatibility.html)
- [ONNX Runtime On-Device Training](https://onnxruntime.ai/docs/api/python/on_device_training/overview.html)


## Jetson Orin Nano

- JetPack: 6.0
- CUDA: 12.2.12

ONNX Runtime build command:

```shell
./build.sh --config RelWithDebInfo --update --build --build_shared_lib --parallel --enable_training --build_wheel --use_cuda --cuda_home /usr/local/cuda --cuda_version=12.2 --cudnn_home /usr/lib/aarch64-linux-gnu --compile_no_warning_as_error
```

References:
- [Can’t install onnxruntime-training on JetPack 5.1.2 (Jetson Orin)](https://forums.developer.nvidia.com/t/cant-install-onnxruntime-training-on-jetpack-5-1-2-jetson-orin/287593)
- [[Training] [On-device-training] Is it possible to build an onnxruntime-training Python module without onnx and torch deps #18991](https://github.com/microsoft/onnxruntime/issues/18991)
- [[Build] build error with --enable_training #19063](https://github.com/microsoft/onnxruntime/issues/19063)


## Jetson Nano

- JetPack: 4.6.1
- CUDA: 10.2

## Hailo-8 AI Accelerator
