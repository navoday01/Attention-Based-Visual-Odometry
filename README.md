# Attention-Based-Visual-Odometry

In this project, we are developing a novel artificial neural network model that can be used to calculate visual odometry. The proposed model is a temporal-based attention neural network, the model takes in raw pixel and depth values from a camera and uses these inputs to generate feature vectors. The model then stores a history of the generated feature vectors, which allows it to take into account the temporal relationship between the current and past data points. 
The resulting output of the model is a prediction of the odometry value, which represents the movement of the camera or object. 
The proposed model has been tested in a variety of scenarios and has shown robust performance in unknown and cluttered environments.

## 💽 Dataset
### 7Scenes
The 7-Scenes dataset is a collection of tracked RGB-D camera frames.

 ![Alt text](assets/7-scenes-7-scenes-overview.png)
 :--:
  *7Scenes* 

### NYU sparse dataset
Similarly, NYU sparse dataset is a collection of tracked RGB-D camera frames in featureless environment of 5 Metrotech.
 ![Alt text](assets/NYC-sparse-dataset.png)
 :--:
  *NYU Sparse Dataset* 

## ⚙️ Setup

1. Install all the requirements (PyTorch v1.13.0)
```shell
pip3 install torch torchvision torchaudio torchinfo tensorboard
```
2. Clone the GitHub repository
```shell
git clone https://github.com/navoday01/ResNet5M-CIFAR10.git
```

## 🏁 Tenserboard

## ⏳ Training
Run train script `training.py` to recreate similar model
```shell
python3 training.py
```
## 🖼 Testing

 To Reproduce the accuracy of the model, run `test.py` and **ensure the model is on the right folder and change the path in the file.** This script will normalise the images to right value.
```shell
python3 test.py
```


## 📊 Results
| Sr. No.|    Model Name    |  # Conv Channels at layer  |  Optimizer  |  Params  |  Test Acc  |  File Link  |
|--------|------------------|----------------------------|------------ |----------|-------------|-------------|
|   1    |  Resnet20SGD     |     [64,128,192,256]       |    SGD+M    |   4.47M  |  93.32%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet20SGD.pt)            |
|   2    |  Resnet20AdaGrad |     [64,128,192,256]       |    Adagrad  |   4.47M  |  90.55%     | -            |
|   3    |  Resnet20RMSprop |     [64,128,192,256]       |    RMSProp  |   4.47M  |  89.13%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetRMSProp.pt)                    |
|   4    |  Resnet20Adam    |     [64,128,192,256]       |    Adam     |   4.47M  |  93.05%     |  -          |
|   5    |  Resnet18Adam    |     [64, 128, 232, 268]    |    Adam     |   4.99M  |  81.03%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet18Adam.pt)                   |
|   **6**    |  **Resnet18SGD**     |     **[64, 128, 232, 268]**    |    **SGD+M**   |   **4.99M**  |  **95.55%**     | [**LINK**](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet18SGD.pt)                     |
|   7    |  Resnet18Bn      |     [64, 118, 178, 256]    |    SGD+M    |   4.99M  |  91.97%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetBn.pt)                   |
|   8    |  Resnet18BnBb    |     [64, 128, 232, 256]    |    SGD+M    |   4.99M  |  92.39%     |   [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetBnBb.pt)                  |
|   9    |  Resnet18BbBn    |     [64, 100, 160, 256]    |    SGD+M    |   4.99M  |  92.73%     |      -       |
|   10   |  Resnet56SGD     |     [64, 96, 128, 190]     |    SGD+M    |   4.98M  |  95.51%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet56SGD.pt)                   |
|   11   |  Resnet56Adam    |     [64, 128, 232, 268]    |    Adam     |   4.98M  |  93.37%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet56Adam.pt)                    |
|   12   |  Resnet156       |     [64, 72, 74, 90]       |    SGD+M    |   4.99M  |  93.82%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet156.pt)                   |

## 📦 Conclusion

A temporal attention (TA) network was created to calculate the visual odometry of the most recent frame using a set of previous frames.

A new dataset called the NYU sparse dataset was created for testing the performance of a visual odometry algorithm in sparse feature environments. The network being tested showed promising results on this dataset and a separate dataset as well. The model's robust performance can be attributed to the fact that the Depth Maps are not being computed but metric Depth Map are taken directly from the dataset, doing this eliminates the scale requirement as it is already encoded in the Depth Map. 

The intent of this project was to create a TA network which was accomplished, further training and testing was conducted on a dataset used in VO benchmarks. Our other goal was to test the model against other state of the art algorithms, to see where it stands. Additionally, the last deliverable was to create a sparse featureless dataset which was also accomplished successfully.

## 👩‍⚖️ Acknowledgement

We express our gratitude to the all the authors, who have worked hard to engineer such effective solutions for visual odometry. We would like to thank everyone whose comments and suggestions helped us with the project. We express our sincere gratitude to Professors Chinmay Hegde, Arsalan Mosenia, and the teaching assistant Teal Witter. 
