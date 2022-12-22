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
pip3 install torch torchvision torchaudio torchinfo tensorboard matplotlib numpy pandas opencv-python 
```
2. Clone the GitHub repository
```shell
git clone https://github.com/navoday01/ResNet5M-CIFAR10.git
```

## 📉 Tensorboard
To view the losses use Tensorboard to visualize the graphs
```shell
tensorboard --logdir runs/
```

## ⏳ Training
Run train script `training.py` to recreate similar model.
```shell
python3 training.py
```
## 🖼 Testing

 To test the model, run `test.py` and the predicited camera poses along with the actual camera poses get saved as a .csv file. The .csv file can be used to vizualize the camera trajectory.
```shell
python3 test.py
```


## 📊 Results

 ![Alt text](assets/Plots.png)
 :--:
  *Losses* 




| **Scene** |   **SCoRe Forest**    |  **DSAC**  |  **VLocNet++**  |  **NNnet**  |  **PoseNet2**  |  **Ours**  |
|--------|------------------|----------------------------|------------ |----------|-------------|-------------|
|  Chess |  0.03m, **0.66°**     |     [64,128,192,256]       |    SGD+M    |   4.47M  |  93.32%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet20SGD.pt)            |
|  Fire  |  Resnet20RMSprop |     [64,128,192,256]       |    RMSProp  |   4.47M  |  89.13%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetRMSProp.pt)                    |
|  Heads |  Resnet20Adam    |     [64,128,192,256]       |    Adam     |   4.47M  |  93.05%     |  -          |
| Office |  Resnet18Adam    |     [64, 128, 232, 268]    |    Adam     |   4.99M  |  81.03%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet18Adam.pt)                   |
| Pumpkin|  **Resnet18SGD**     |     **[64, 128, 232, 268]**    |    **SGD+M**   |   **4.99M**  |  **95.55%**     | [**LINK**](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet18SGD.pt)                     |
|RedKitchen|  Resnet18Bn      |     [64, 118, 178, 256]    |    SGD+M    |   4.99M  |  91.97%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetBn.pt)                   |
|  Stairs  |  Resnet18Bn      |     [64, 118, 178, 256]    |    SGD+M    |   4.99M  |  91.97%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetBn.pt)                   |

## 📦 Conclusion

A temporal attention (TA) network was created to calculate the visual odometry of the most recent frame using a set of previous frames.

A new dataset called the NYU sparse dataset was created for testing the performance of a visual odometry algorithm in sparse feature environments. The network being tested showed promising results on this dataset and a separate dataset as well. The model's robust performance can be attributed to the fact that the Depth Maps are not being computed but metric Depth Map are taken directly from the dataset, doing this eliminates the scale requirement as it is already encoded in the Depth Map. 

The intent of this project was to create a TA network which was accomplished, further training and testing was conducted on a dataset used in VO benchmarks. Our other goal was to test the model against other state of the art algorithms, to see where it stands. Additionally, the last deliverable was to create a sparse featureless dataset which was also accomplished successfully.

## 👩‍⚖️ Acknowledgement

We express our gratitude to the all the authors, who have worked hard to engineer such effective solutions for visual odometry. We would like to thank everyone whose comments and suggestions helped us with the project. We express our sincere gratitude to Professors Chinmay Hegde, Arsalan Mosenia, and the teaching assistant Teal Witter. 
