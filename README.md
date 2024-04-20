# Accelerated Continual Learning

Requirements:
- `onnxruntime==1.16.3`

References:
- [ONNX opset support](https://onnxruntime.ai/docs/reference/compatibility.html)
- [ONNX Runtime On-Device Training](https://onnxruntime.ai/docs/api/python/on_device_training/overview.html)

To simplify project reproduction, Python wheels for utilized libraries were created and are available [here](https://chmura.put.poznan.pl/s/L7qZOvJ9375lSMC).

## Jetson Orin Nano

- JetPack: 6.0
- CUDA: 12.2.12

### ONNX Runtime

#### Build command:

```shell
./build.sh --config RelWithDebInfo --update --build --build_shared_lib --parallel --enable_training --build_wheel --use_cuda --cuda_home /usr/local/cuda --cuda_version=12.2 --cudnn_home /usr/lib/aarch64-linux-gnu --compile_no_warning_as_error
```

#### References:
- [Can’t install onnxruntime-training on JetPack 5.1.2 (Jetson Orin)](https://forums.developer.nvidia.com/t/cant-install-onnxruntime-training-on-jetpack-5-1-2-jetson-orin/287593)
- [[Training] [On-device-training] Is it possible to build an onnxruntime-training Python module without onnx and torch deps #18991](https://github.com/microsoft/onnxruntime/issues/18991)
- [[Build] build error with --enable_training #19063](https://github.com/microsoft/onnxruntime/issues/19063)


### PyTorch

Python PyTorch v2.2.0 wheel for JetPack 6.0 with Python 3.10 was installed from [Jetson zoo](https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29) according to [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) instruction. _torchvision_ was built from source with regards to this instruction (_Instructions -> Installation -> torchvision_).

## Jetson Nano

- JetPack: 4.6.1
- CUDA: 10.2

### ONNX Runtime

#### Requirements:
1. Customized version of [onnxruntime](https://github.com/MatPiech/onnxruntime) with v1.16.3 tag and CUDA 10.2 support.
2. gcc-9 and g++-9 installed and selected as compilers.
3. Updated CUDA header files (`crt/host_config.h`) and `/usr/include/c++/9/bits/stl_function.h` according to [CUDA 10.2 incompatible with GCC 9.3 and Clang 9.0](https://github.com/espressomd/espresso/issues/3654#issuecomment-612165048) issue comment.

#### Build command:

```shell
./build.sh --config RelWithDebInfo --update --build --build_shared_lib --parallel --enable_training --build_wheel --use_cuda --cuda_home /usr/local/cuda --cuda_version=10.2 --cudnn_home /usr/lib/aarch64-linux-gnu --compile_no_warning_as_error
```

#### References:
- [CUDA 10.2 incompatible with GCC 9.3 and Clang 9.0](https://github.com/espressomd/espresso/issues/3654#issuecomment-612165048)

## Hailo-8 AI Accelerator
