# YOLOv8 目标检测专用源码

基于官方 https://github.com/ultralytics/ultralytics 的 v8.0.100，
提交 dce4efce48a05e028e6ec430045431c242e52484。官方远端不存在 v8.0.0 标签。
这是经过裁剪的本地版本，不是未经修改的官方发布版本。许可证见 LICENSE。

保留标准 YOLOv8 n/s/m/l/x 检测结构、网络模块、训练、验证、推理、导出、数据加载和共用工具。
移除其他架构、其他 YOLO 版本配置，以及分割、分类、姿态任务入口和模型类。
共用数据处理、可视化和后端工具可能仍包含非检测任务的通用分支。
为现代 PyTorch 加载官方完整模型权重显式使用 weights_only=False；仅加载可信权重。

## 路径

- 权重：yolov8n.pt
- 模型结构：ultralytics/models/v8/yolov8.yaml
- 模型构建：ultralytics/nn/tasks.py
- 网络层与检测头：ultralytics/nn/modules/
- 训练/推理/验证：ultralytics/yolo/v8/detect/
- 训练与推理引擎、导出：ultralytics/yolo/engine/
- 数据集配置：ultralytics/datasets/

## 使用

```bash
conda activate yolov8
cd /sda/zhangchuyi/yolov8
python -m pip install -e .
yolo detect predict model=yolov8n.pt source=ultralytics/assets/bus.jpg device=cpu
yolo detect train model=yolov8n.pt data=/absolute/path/to/data.yaml epochs=100 imgsz=640
```

整理时已验证 CPU 权重推理、从 YAML 构建网络、检测损失与反向传播。
未验证完整数据集训练或所有导出格式。
