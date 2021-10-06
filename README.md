# Yolor-ROS
A set of python scripts implementing Yolor with TensorRT for object detection in ROS.

## Preparation
1. Clone this resipiratory to your catkin workspace
```
cd /catkin_ws/src
git clone https://github.com/petpetpeter/Yolor-ROS.git
```
2. Build your workspace
```
cd /catkin_ws/
catkin build
```
3. Install Python Dependencies
```
pip install -e requirements.txt
```
4. Launch ROS Yolor example
```
roslaunch yoloros camera.launch
```

> TensorRT 8.0.3.4: Follow installation guide here https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
1. Convert pytorch weight file to onnx file
```
roscd yoloros/scripts/yolor
python convert2Onnx.py --weight weights/yolor_p6.pt --out onnx/yolor_p6.onnx 
```
2. Convert onnx to tensorRT Engine
```
/usr/src/tensorrt/bin/trtexec --explicitBatch \
                                --onnx=transformer_encoder.onnx \
                                --saveEngine=transformer_encoder.trt \
                                --minShapes=input:1x3x32x128 \
                                --optShapes=input:32x3x32x512 \
                                --maxShapes=input:32x3x32x768 \
                                --verbose \
                                --fp16
```
3. Launch ROS Yolor TensorRT example
```
roslaunch yoloros trt_camera.launch
```

## Performance
Yolor | Baseline-FP16 | TensorRT-FP16 
--- | --- | ---  
Inference Time | 94.8 ms | 43.4 ms 
GPU Memory | 2480 Mb | 1613 Mb

## Acknowledgement
- Yolor: https://github.com/WongKinYiu/yolor
- Yolor-TensorRT: https://github.com/NNDam/yolor
