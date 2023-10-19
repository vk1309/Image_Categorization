# MRI Image Categorization

## 1. Introduction 

A tumor results from an uncontrolled division of abnormal cells forming a mass that can hamper the normal functionality of the tissue or organ. Tumors may be malignant (high-grade) which are cancerous or benign (low-grade). Compared to a benign brain tumor, a malignant brain tumor grows very rapidly and is more prone to invade adjacent tissues. Thus, a primary malignant brain tumor has a dreary prognosis and considerably reduces cognitive function and quality of life.Cancers pertaining to the brain and various nervous system rank tenth in the leading causes of mortality and is considered as the 3rd most prevalent cancer among teenagers and adults. The causes of a brain tumor can be attributed to environmental aspects such as excessive usage of artificial chemicals or genetic factors. Treatment options include radiotherapy, chemotherapy, and surgical procedure. The earlier a brain tumor is detected the higher the chances of survival and the wider the treatment options. Various methods may be employed to diagnose a brain tumor such as MRI scan, BIOPSY and SPECT (Single Photon Emission Computed Tomography) scan.

Magnetic Resonance Imaging (MRI) is the most prevalent method due to its non-intrusive imaging modality that provides distinctive tissue contrast. It is also highly conformable in terms of normalization of tissue contrast providing minute details of interest. However, limitations are faced because of the uncertain, random, and irregular size, shape, and locality of brain tumors. Also, non-autonomous partitioning of the tumors comes at the cost of a lot of time and is a labor-intensive, cumbersome, and largely subjective task given the amount of data to be handled thus reducing accuracy.

The algorithms in this paper applied for segmentation of brain tumor can be categorized as traditional and techniques pertaining to deep learning. The former such as Agglomerative Clustering, Principal Component Analysis (PCA) and Support Vector Machines (SVM) and for the later we implemented ResNet50 Neural Network model with no preprocessing requirements by inputting the MRI image into a two-dimensional CNN model. The output is a brain tumor classification whether the patient has a Cancerous tumor or not. However, the number of images used in this method for the training process is relatively less raising acceptability queries for datasets with a large number of images.
The overall paper is laid out in the following manner: Section II provides a quick summary of the proposed method. Section III gives details about related work and implementations Section IV details of the methodology propositioned and the dataset and analysis details. Section V deals with the experimental results, performance evaluation, and the result comparison for the models outlined in Section IV. Finally, section VI pertains to the conclusion

## 2. About the Dataset 

The datast used for the project includes 253 MRI images of the brain obtained from kaggle. This dataset includes both benign (155) and malignant (98) tumor instances with a resolution of 256x256 pixels.For this project, the images were preprocessed and labeled to emphasize the tumor areas. The challenges such as class imbalance, variable image clarity, and differences in tumor size and shape were identified.

Gaussian Blur was applied on all the images to brighten the images and reduce noise before cropping the images. Post this, four contours (left, right, top and bottom) were identified using which the original images were cropped to form new images. Finally, Principal Component Analysis was applied to extract 51 features from a total of 256 features, retaining 98% variance.

<img width="472" alt="Screenshot 2023-10-19 at 2 47 41 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/4a2ff3fa-b0d8-433c-b881-81e4de54d4d1">

Finally, data augmentation was performed to increase the size and diversity of a training dataset by artificially creating new data samples from existing ones. This was achieved by applying various transformations like rotations, translations, scaling, flipping, cropping, and adding noise or distortions to original data. This was done to expose the model to more variations of the same data. It helped reduce overfitting and improve the generalization performance of the model on unseen data. The augment images are depicted below, where the first image is our original image, and the ones after that are a result of data augmentation. The code for Data Augmentation is available in the 5230_Aglomerative_&_feature_selection notebook line 7.

<img width="632" alt="Screenshot 2023-10-19 at 2 48 38 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/db8e9360-de5a-46fe-901b-fbbf60db1c64">

## 3. Methods

1) Support Vector Machine (Non - Linear):

The image segmentation for brain tumor prognosis using support vector machine (SVM) was performed using the following methodology: First, the images were resized to a standard size of (200,200) pixels. The pixel values were then normalized to fall within the range of 0 and 1. This ensured that the input features were in a uniform range and that the SVM could accurately classify the images.

Principal component analysis (PCA) was applied to reduce the computational cost of the algorithm. This technique reduced the dimensionality of the feature space while retaining most of the original information. This significantly reduced the training time and improved the accuracy of the SVM.
To optimize the hyperparameters of the SVM, GridSearchCV was used. The best parameters observed were {'C': 5, 'gamma': 0.0001, 'kernel': 'rbf'}. It was noted that SVM worked well on non-linear data and could accurately classify images with a low amount of noise in the dataset. However, it was not suitable for datasets with high levels of noise, as this could affect the accuracy of the classifier.

The performance of the SVM was evaluated on a testing dataset. It was observed that 18 misclassified samples were present in the testing data of size 51. The performance was carefully analyzed to identify any potential areas of improvement in the methodology. Overall, the SVM approach showed decent results for image segmentation in brain tumor prognosis.

<img width="923" alt="Screenshot 2023-10-19 at 2 50 59 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/4761066f-6983-4012-a4c1-f22831dcc64e">

All of the kernels have one common parameter called cost (C). This parameter is the constant of constraint violation that observes whether a data sample is classified on the wrong side of the decision limit[9]. The specific parameters to tune up each kernel are detailed herewith. We have used the RBF kernel for our model. The width of the Gaussian radial basis function can be adjusted by the parameter γ.

Below is the classification report obtained on the SVM model:

<img width="548" alt="Screenshot 2023-10-19 at 2 52 07 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/0465454b-72fc-4bd7-a55b-1b821de6d885">

From the classification report it is seen that the precision, recall and f1-score for class ‘0’ belonging to the benign cases are comparatively low. This could happen because of the lower support in opposition to the cases that are malignant. The training and testing scores are 0.935 and 0.627 respectively.

2) Agglomerative hierarchical clustering:
   
Agglomerative hierarchical clustering is a popular method for grouping similar objects or data points into clusters. The algorithm starts with each data point in its own cluster and then iteratively merges the closest pairs of clusters until all data points are in a single cluster. In this way, the algorithm builds a hierarchical structure of nested clusters, with smaller clusters nested within larger ones.

There are several different linkage criteria that can be used to define the distance between clusters, such as single linkage, complete linkage, average linkage, and Ward's linkage. The formula for Ward's linkage is: d(C1,C2) = √{[n1/(n1+n2)]*D(C1) + [n2/(n1+n2)]D(C2) + [(n1n2)/((n1+n2)^2)]*D(C1,C2)}; where C1 and C2 are two clusters, n1 and n2 are the number of points in the two clusters, D(C1) is the variance of the distances between points in cluster C1, D(C2) is the variance of the distances between points in cluster C2, D(C1,C2) is the variance of the distances between points in C1 and points in C2, and √{} denotes the square root.

Ward's linkage is a popular choice for clustering data with continuous variables, because it minimizes the sum of squared differences within each cluster. This criterion tends to produce more compact, spherical clusters of similar size and density, which are often easier to interpret than clusters generated by other linkage methods. In our project, wards linkage generates the highest accuracy, and the final accuracy of the testing set is only 55%.

The potential reason for this poor performance is that the agglomerative clustering model was not able to capture the complex relationships between the features and the classes, leading to poor performance. Therefore, the F1 score of 0.5, precision of 0.75, and recall of 0.375 indicate that the agglomerative hierarchical clustering model has correctly identified some positive samples but is not performing well overall. It has a higher precision than recall, which means that it tends to have few false positives but misses a significant number of actual positives. The model may require further optimization or the use of a different method to improve its performance.

<img width="479" alt="Screenshot 2023-10-19 at 2 53 41 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/2c0a169b-f66d-4663-9488-2aa8b2eff95b">

3) ResNet50 CNN:
The ResNet50 architecture implemented in this paper is based on the research paper[10].The model architecture consists of five stages, each containing multiple convolutional layers. The input image is initially passed through a convolutional layer with 64 filters, a kernel size of 7x7, and a stride of 2, which reduces the spatial dimensions of the input by a factor of 2. This is followed by a max pooling layer with a kernel size of 3x3 and a stride of 2, which further reduces the spatial dimensions.

The five stages of ResNet50 consist of blocks of convolutional layers, each block containing multiple layers. The first stage has three blocks, each with 64 filters of size 1x1, 64 filters of size 3x3, and 256 filters of size 1x1, respectively. The second stage has four blocks, each with 128 filters of size 1x1, 128 filters of size 3x3, and 512 filters of size 1x1, respectively. The third stage has six blocks, each with 256 filters of size 1x1, 256 filters of size 3x3, and 1024 filters of size 1x1, respectively. The fourth stage has eight blocks, each with 512 filters of size 1x1, 512 filters of size 3x3, and 2048 filters of size 1x1, respectively. Finally, the fifth stage consists of a single global average pooling layer followed by a fully connected layer with 1000 units, corresponding to the 1000 classes in the ImageNet dataset.

The skip connections in ResNet50 are implemented using identity mappings, where the input of a block is added directly to the output of the block. This allows the gradients to flow through the skip connections and enables the training of very deep neural networks.

Magnetic resonance images are rich in details. The resulting feature map can be represented as follows:

<img width="985" alt="Screenshot 2023-10-19 at 2 54 43 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/99d78d48-6cf6-4390-b2a0-7e1e3dcdda4f">

<img width="511" alt="Screenshot 2023-10-19 at 2 55 11 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/4e257995-a85b-4b9a-8c6a-fe94d4a2e444">


## 4. Results

Our experiments show that data augmentation is an effective technique for improving the accuracy of the CNN model. The results of the experiments indicate that the model's accuracy is significantly improved when augmented data is included in the training set. The limitation of this study is the limited size of data (253 MRI images) which could negatively influence the predictive power of the models. Variable image clarity, differences in tumor size and form are all possible problems with the data and potential overfitting problems.
Among the three models implemented, CNN generates the highest accuracy on the testing set. The CNN model achieves an accuracy of 86.9% on the testing set, while the SVM and agglomerative hierarchical clustering models achieve 64.7% and 55%, respectively. This result suggests that the CNN model is more effective in identifying and classifying brain tumors from MRI images.

<img width="667" alt="Screenshot 2023-10-19 at 2 56 36 PM" src="https://github.com/vk1309/Image_Categorization/assets/39329373/480fc618-44b4-4d5d-9e76-4cca452a51b7">

Although the CNN model showed superior performance compared to SVM and agglomerative hierarchical clustering, future work could include making this a scalable model. The SVM model has comparatively lower training runtime in comparison to hierarchical clustering and CNN, making it more suitable for larger datasets. However, the CNN model generated the highest accuracy and with the shortest testing runtime, making it more suitable for real-time applications where quick and accurate diagnoses are required.

However, there is still room for improvement in the CNN model. One way to improve the model in the future work is by using transfer learning. Transfer learning is a technique where a pre-trained model is used as a starting point for a new task. This approach can save a lot of time and resources as the model does not need to be trained from scratch. Instead, the pre-trained model is fine-tuned on the new dataset. This can lead to better performance as the pre-trained model has already learned useful features from a large dataset.

## 5. References

[1] Huang, Y., et al. (2019). Brain tumor prognosis analysis using convolutional neural network and k-means clustering. International Conference on Artificial Intelligence and Computer Engineering.

[2] Zhang, Z., et al. (2020). Brain tumor prognosis prediction using a recurrent neural network based on multimodal MRI images. Neural Computing and Applications.

[3] Li, L., et al. (2020). Brain tumor classification using support vector machine based on MRI features. Journal of Medical Imaging and Health Informatics.

[4] Kim, H., et al. (2021). Predicting brain tumor patient survival using convolutional neural networks and random forest. Journal of Digital Imaging.

[5] Chen, L., et al. (2021). Deep learning-based classification and prognostication of brain tumors using both imaging and genomic data. IEEE Transactions on Medical Imaging.

[6] “Brain MRI Images for Brain Tumor Detection”, kaggle.com. https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection (accessed Apr.23, 2019)

[7]“Brain Tumor Detection CNN”, kaggle.com. https://www.kaggle.com/code/loaiabdalslam/brain-tumor-detection-cnn (accessed Apr. 1, 2019)

[8]“Brain Tumor DetectionDenseNet”, kaggle.com. https://www.kaggle.com/code/sevvalbicer/brain-tumor-detection-densenet (accessed May. 23, 2021)

[9] Himar Fabelo, et al. (2018). SVM Optimization for Brain Tumor Identification Using Infrared Spectroscopic Samples, NIH Publication.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015). ResNet50 Model Architecture: Deep Residual Learning for Image Recognition (arXiv:1512.03385)















