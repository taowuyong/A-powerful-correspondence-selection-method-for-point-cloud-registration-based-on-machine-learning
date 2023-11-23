# A-powerful-correspondence-selection-method-for-point-cloud-registration-based-on-machine-learning
In this project, we provide the Matlab code of a correspondence selection method based machine learning. The method uses two geometric constraints to construct a feature vector for each correspondence. Then, the feature vectors are input into a trained support vector machine (SVM) classifier to predict the labels of the correspondences.

Everyone is welcome to use the code for research work, but not for commerce. If you use the code, please cite my paper (Wuyong Tao, Dong Xu, Xijiang Chen, Ge Tan, (2023). A powerful correspondence selection method for point cloud registration based on machine learning. Photogrammetric Engineering & Remote Sensing. 89 (11): 703-712)

In this project, four files are provided. The "LRF_TriLCI" file is used to calculate the LRF of keypoint. The "TriLCI" file is used to calculate the TriLCI descriptor. The “svm,mat” file is the trained SVM. The “correspondence selection method based SVM” shows how to use the correspondence selection method to select correspondences and perform point cloud registration. 

Before you carry out our algorithm, you need to calculate the point cloud resolution (pr).
