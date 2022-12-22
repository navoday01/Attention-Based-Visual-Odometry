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
Similarly, NYU sparse dataset is a collection of tracked RGB-D camera frames.
 ![Alt text](assets/7-scenes-7-scenes-overview.png)
 :--:
  *7Scenes* 
