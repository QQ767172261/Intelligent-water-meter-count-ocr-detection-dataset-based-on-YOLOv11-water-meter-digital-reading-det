基于 YOLOv11 水表数字读数检测系统  智能水表计数ocr检测数据集，9776张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8fed264d919f4f8f8747fc91d5877f00.png)

图像尺寸:416*416
类别数量:10类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e8d25d61b90478f927ca8304993b95c.png)

训练集图像数量:7506; 验证集图像数量:1674； 测试集图像数量:596
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6ac7093fab4c473ba6ddcf44d3d09d25.png)

3: 3602,4378
9: 3221,3754
1: 5151,6802
0: 9149,16701
4: 3613,4333
2: 4202,5277
5: 3379,4057
7: 3179,3815
8: 3212,3733
6: 3356,3939
image num: 9776

模型代码：提供全部训练及测试源代码，模型训练使用yolov11n训练，50个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6b63ae658d504bc1b2ca4cf3e9e809c4.png)

qt界面：运行界面采用pyqt5编写， 提供全部源代码，支持图片、视频及摄像头进行检测: 界面可实时显示目标位置、目标总数、置信度等信息: 支持检测结果保存;本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c97e943d33294447960b4ca4dba27d77.png)



# 基于YOLOv11水表数字读数检测系统
## 环境安装
```bash
pip install ultralytics opencv-python pyqt5 numpy pandas
```

## 数据集配置 water_meter.yaml
```yaml
path: ./water_meter_dataset
train: images/train
val: images/val
test: images/test

names:
  0: 0
  1: 1
  2: 2
  3: 3
  4: 4
  5: 5
  6: 6
  7: 7
  8: 8
  9: 9
```

## 训练脚本 train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="water_meter.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=False,
        project="runs/train",
        name="water_meter_det"
    )
    print("训练完成！")
    metrics = model.val()
    print(f"mAP@0.5:{metrics.box.map50}")
```

## 推理脚本 predict.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c87188d35a274acfb6353e7f28d2591f.png)

```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/train/water_meter_det/weights/best.pt")

def predict_image(img_path):
    res = model(img_path, conf=0.3)
    for r in res:
        img = r.plot()
        cv2.imwrite("water_meter_result.jpg", img)
        print(f"检测到数字数量：{len(r.boxes)}")
    return img

predict_image("test.jpg")
```

## PyQt5可视化界面 main_ui.py
```python
import sys
import cv2
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QWidget,
                             QHBoxLayout, QVBoxLayout, QTableWidget, QTableWidgetItem, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

model = YOLO("runs/train/water_meter_det/weights/best.pt")

class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame, conf=0.3)
            frame = res[0].plot()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.change_pixmap.emit(qimg)
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的水表计数检测系统")
        self.setGeometry(100,100,1250,900)
        self.thread = None
        self.result_data = []

        w = QWidget()
        self.setCentralWidget(w)
        main_layout = QHBoxLayout(w)

        # 左侧图片+结果表格区域
        left_layout = QVBoxLayout()
        self.img_label = QLabel("图像展示区域")
        self.img_label.setFixedSize(720,520)
        left_layout.addWidget(self.img_label)

        # 检测结果表格
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号","文件路径","类别","置信度","坐标位置"])
        left_layout.addWidget(self.table)

        # 右侧控制面板
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("文件导入"))
        self.btn_img = QPushButton("选择图片")
        self.btn_video = QPushButton("选择视频文件")
        self.btn_cam = QPushButton("开启摄像头")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_cam.clicked.connect(self.open_cam)
        right_layout.addWidget(self.btn_img)
        right_layout.addWidget(self.btn_video)
        right_layout.addWidget(self.btn_cam)

        right_layout.addWidget(QLabel("检测结果"))
        self.label_time = QLabel("用时：-- s  目标数目：--")
        self.combo_target = QComboBox()
        self.combo_target.addItem("全部")
        self.label_conf = QLabel("置信度 -- %")
        self.label_pos = QLabel("xmin: -- ymin: --\nxmax: -- ymax: --")
        right_layout.addWidget(self.label_time)
        right_layout.addWidget(self.combo_target)
        right_layout.addWidget(self.label_conf)
        right_layout.addWidget(self.label_pos)

        # 操作按钮
        self.btn_save = QPushButton("保存")
        self.btn_exit = QPushButton("退出")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_exit.clicked.connect(self.close)
        right_layout.addWidget(self.btn_save)
        right_layout.addWidget(self.btn_exit)

        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_widget)

    def load_img(self):
        file,_ = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;*.png;*.bmp")
        if not file:
            return
        src = cv2.imread(file)
        res = model(file,conf=0.3)[0]
        img_res = res.plot()
        rgb_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_res.shape
        q_res = QImage(rgb_res.data,w,h,ch*w,QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(q_res))

        boxes = res.boxes
        self.result_data.clear()
        self.table.setRowCount(0)
        for idx, box in enumerate(boxes):
            conf = float(box.conf)
            cls = res.names[int(box.cls)]
            xyxy = box.xyxy.tolist()[0]
            x1,y1,x2,y2 = map(int,xyxy)
            self.result_data.append([idx+1, file, cls, f"{conf:.2f}", f"[{x1},{y1},{x2},{y2}]"])
            self.table.insertRow(idx)
            self.table.setItem(idx,0,QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1,QTableWidgetItem(file))
            self.table.setItem(idx,2,QTableWidgetItem(cls))
            self.table.setItem(idx,3,QTableWidgetItem(f"{conf:.4f}"))
            self.table.setItem(idx,4,QTableWidgetItem(f"[{x1},{y1},{x2},{y2}]"))
        if len(boxes)>0:
            self.label_time.setText(f"用时：{res.speed['inference']/1000:.3f} s  目标数目：{len(boxes)}")
            box0 = boxes[0]
            self.label_conf.setText(f"置信度{float(box0.conf)*100:.2f} %")
            x1,y1,x2,y2 = map(int,box0.xyxy.tolist()[0])
            self.label_pos.setText(f"xmin: {x1}   ymin: {y1}\nxmax: {x2}   ymax: {y2}")

    def load_video(self):
        path,_ = QFileDialog.getOpenFileName(self,"打开视频","","*.mp4;*.avi")
        if path:
            model.predict(path,save=True,conf=0.3)

    def open_cam(self):
        self.thread = VideoThread()
        self.thread.change_pixmap.connect(lambda img:self.img_label.setPixmap(QPixmap.fromImage(img)))
        self.thread.start()

    def save_result(self):
        df = pd.DataFrame(self.result_data,columns=["序号","文件路径","类别","置信度","坐标位置"])
        df.to_csv("water_meter_det_result.csv",index=False,encoding="utf-8-sig")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
```
运行：`python main_ui.py`
功能：图片/视频/摄像头读取水表数字，检测0-9共10类数字，输出检测框、类别数字、置信度、坐标，表格展示，结果导出csv，
