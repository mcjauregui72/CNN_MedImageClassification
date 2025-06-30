# Medical Image Classification with Convolutional Neural Networks: Pre-trained and Custom Models Evaluated Individually, as an Ensemble, and Chained


## Executive Summary

Can we train a custom convolutional neural network (CNN) model from scratch to classify CT chest scan images with one of the following four conditions: Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, or normal cells? How well does the model perform? 
  
What happens if we modify a pre-trained model for the same classification task? How well does that model perform? What happens if we add the pre-trained model to the custom CNN by employing model ensembling and transfer learning? Will we see improved accuracy scores with either of these methods?

We trained and evaluated four models  
 a) first_model, a pre-trained ResNet50-based model  
 b) second_model, a custom CNN  
 c) ensemble_model, which ensembled the predictions of model_one and model_two by averaging them  
 d) chained_model, an example of transfer learning in which we chained first_model and second_models  
  
We compared the training loss, training accuracy, validation loss, validation accuracy, balance score, average accuracy score, and composite score values among the four models. With decent learning on the training data, mild overfitting, and the best generalization of all four, the chained_model demonstrated the best overall performance. It also achieved the highest composite score, indicating it was the most balanced and accurate of the models. Our recommendation was to use the chained_model as the preferred choice of models for the task at hand.

Concepts discussed:  
Convolutional Neural Networks  
Pre-trained models  
Model Ensembling  
Transfer Learning/Model Chaining  
Model Loss  
Model Accuracy  
Model Balance Score  
Model Average Accuracy Score  
Model Composite Score  
  
The data for this project was obtained here: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images. The notebook for this project is SubmodelsEnsembleChain.ipynb.

## Project Overview

Data consisted of CT scan images revealing the presence of one of three chest carcinomas or the presence of healthy cells. We tasked four models with classifying each image as one of the following four outcomes: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, or healthy cells. All images were divided into training (613 images), testing (315 images), and validation (72 images) sets containing the four classes of images.

We trained two convolutional neural network (CNN) models to classify CT chest scan images, a pre-trained ResNet50-based model and a custom CNN. We next investigated the benefit of using the pre-trained model in conjunction with the custom CNN by employing model ensembling and transfer learning (chaining). 

CNNs use convolutional and pooling layers to automatically and hierarchically learn features from images, followed by fully connected layers to classify those features into predefined categories. This process enables CNNs to handle and classify complex visual data.  

In this project, we used the ResNet50 CNN, trained to classify 14 million images in the ImageNet dataset into 1,000 different categories, as the base of our pre-trained model. This entailed modifying the ResNet50 to make it compatible for both ensembling and chaining with the custom CNN. 

Because we were working with four distinct image labels classes (adenocarcinoma, large cell carcinoma, squamous cell carcinoma, and healthy cells), we chose the sparse categorical crossentropy loss function. We anticipated labels as integers because we used the tf.keras.preprocessing.image_dataset_from_directory' function from TensorFlow’s Keras API to load image data from the train, test, and validate directories, which automatically labels images based on directory structure.
We defined, compiled, and trained the individual CNN submodels (pre-trained and custom) on the image data before ensembling and chaining the submodels. We looked for a noticeable improvement in accuracy between the ensembled model and/or the chained model, over each of the two submodels.  
    
   
## Convolutional Neural Networks  
  
CNNs use convolutional and pooling layers to automatically and hierarchically learn features from images, and use fully connected layers to classify those features into predefined categories. This process enables CNNs to effectively handle and classify complex visual data.  
    
We built the ResNet50-based pre-trained model (first_model) and the custom CNN (second_model) with the following components:  
  
1. Input Layer: Where our input images are represented as matrices of pixel values.   

2. Convolutional Layers: These layers applied convolutional filters (or kernels) to the inputs. Each filter scanned the images and performed a convolution operation involving element-wise multiplication and results summing. These layers extracted features like edges, textures, and patterns from each image and produced a feature map highlighting the presence of specific features in different parts of the image. 
  
3. Activation Function: We applied activation function ReLU (Rectified Linear Unit) to introduce non-linearity into the model, which helped the network learn more complex patterns.  
  
4. Pooling Layers: We used max pooling to reduce the spatial dimensions of the feature maps by taking the maximum value from a subset of the feature map. This reduced the number of parameters and computations, helping the network become more robust to variations in image.  
  
5. Flattening (model_two only): Because the output from the convolutional and pooling layers was a multi-dimensional tensor, we needed to flatten the tensor to a one-dimensional vector before feeding it into the fully connected layers.    
  
6. Fully Connected Layers: Similar to traditional neural networks, where each neuron is connected to every neuron in the previous layer, the fully connected layers combined the features learned by the convolutional and pooling layers to make a final prediction.  
  
7. Output Layer: We chose a softmax function capable of outputting probabilities for each of the four classes, indicating the network's prediction of Adenocarcinoma, Large cell carcinoma, Squamous cell carcinoma, or normal cells.   
    
 
## Pre-trained Models

Pre-training a neural network involves training a model on a large, broad, general-purpose dataset before fine-tuning it on a specific task (a new set of specific, likely previously unseen data). The ResNet50 model is a well-known model that was trained on the ImageNet database, a collection of millions of images classified across 1,000 categories.   

During pre-training, the ResNet50 model learned to identify and extract general features from the input data, such as images' edges, textures, and shapes. These features become broadly useful across new tasks and data domains, even if the new data was never part of the training data.

The benefits of pre-training include improved performance, better generalization, and reduced training time. Pre-training allows a model to leverage knowledge learned from a large and diverse dataset. This accumulated knowledge can lead to better performance on the new task, especially when the new dataset is small or lacks diversity. Training a model from scratch can be computationally expensive and time-consuming. Pre-training on a large dataset and then fine-tuning it can significantly reduce the time required to achieve good performance.

Pre-trained models often generalize better to new tasks because they start with a solid understanding of basic features and patterns, which can help improve accuracy on the new task. Pre-training can be a powerful technique, especially when data are scarce or where training a model from scratch would be impractical given resource constraints.  

  
## Submodel Compatibility Considerations
  
To make our pre-trained, ResNet50-based CNN (first_model) and our custom CNN (second_model) compatible with each other for direct ensembling and transfer learning purposes, we took the following precautions and adaptations to the original ResNet50 model.  
  
1. Used the tf.keras.preprocessing.image_dataset_from_directory method to generate our training_set, testing_set, and validation_set.  
   * This method automatically labeled all images based on their subdirectory names. 
   * It also treated each subdirectory as a class, assigning labels as integers starting from 0.  
    
2. Specified image_size as (224, 224) for first_model because ResNet50-based models expect images of that size. For consistency, we set the image_size to (224, 224) for second_model as well.  
     
3. Specified channels, img_shape, and class_count in first_model to be identical to those in second_model.
   
4. Defined the same data augmentation layers in both submodels and applied data augmentation to both models' input tensors.
  
5. Defined the same rescaling layers in both submodels, and specified the input tensor as the scaled inputs
   
6. Applied data augmentation and rescaling in both submodels, early in the model pipeline.
    * When ensembling two models, it is appropriate to apply data augmentation and rescaling in both submodels.
    * In particular, data augmentation should come before rescaling, right after defining the model's input layer.
    * Data augmentation techniques (e.g., RandomRotation, RandomZoom, RandomFlip) are designed to work on raw pixel values in the 0-255 range.
    * If rescaling is done first, pixel values are converted to 0-1, which could interfere with how certain augmentations are applied.
    * Because the ResNet50-based base_model component of our first_model expected inputs' pixel values to be normalized to a range between 0 and 1, we rescaled the augmented input data before passing it to the ResNet50 layers. 
  
7. Specified include_top = False to effectively remove ResNet50's top layer so we could replace it with one suited for our own task. Because the original ResNet50 model was pre-trained to classify over a million ImageNet images into 1,000 classes, it outputs feature maps when its top layer is removed. 

8. Added the pooling='max' layer to the ResNet50-based model to control the shape of the output tensor and ensure compatibility with the subsequent layers that needed to be added.  
   * The ResNet50 model outputs a 4D tensor after its convolutional layers when include_top=False and no pooling is applied. This tensor could not be fed directly into fully connected Dense layers, which require a 2D input.  
   * Setting pooling='max' applied global max pooling to reduce the spatial dimensions into a single value for each channel, producing a tensor compatible with Dense layers.  
   * Without pooling='max', we would have needed to explicitly add a Flatten layer to convert the 4D tensor to 2D to avoid a shape mismatch error. Though a Flatten layer would have resolved the shape issue, it would generate a larger input size for the Dense layers, increasing the risk of overfitting.  
   * Unlike Flattening, which preserves all spatial information to return a high-dimensional feature vector, global pooling reduces dimensionality.   
  
9. Specified for layer in base_model.layers: layer.trainable = False, to avoid re-training ResNet50's pre-trained knowledge during model training.  
   * Making these layers untrainable preserved the features ResNet50 learned during pre-training, keeping them from becoming over-written during training.  
   * Layer freezing effectively turned ResNet50 into a feature extractor.  
    
10. Built both submodels with the Functional API because it supported more flexibility than the Sequential API. In particular, the Functional API
    * Afforded more flexibility when combining pre-trained models with custom layers or sharing layers between models. 
    * Allowed for explicit definition of the flow of data, enabling fine control over how layers connect and interact.  
    * Supported freezing layers and chaining models.  
    * Handled the complexities involved in ensembling models.  
     
11. Added custom layers on top of the ResNet50-based base to allow the final model to complete our four-class classification task and be ensembled and chained with the other submodel.
    * Both the BatchNormalization and Dropout layers helped improve generalization on unseen data.  
    * The Dense(256, activation='relu') layer learned more complex patterns from the high-level features provided by ResNet50.  
    * These more complex patterns became relevant to our classification task.  
    * The relu activation function supported the custom layers to model more intricate relationships between features.  
    * Dropout(0.25) was intended to prevent overfitting by forcing the model to learn more robust features and preventing it from becoming too reliant on specific neurons.   
  
12. Defined identical output layers in each submodel:  
    * Dense(class_count, activation = 'softmax') to output a probability distribution across the classes (given by class_count).  
    * Each value in the probability distribution corresponded to the predicted probability that the input image belonged to a given class.  
    * We chose the Softmax activation because it can return a probability distribution over three or more classes.  
    
13. Compiled both submodels with optimizer='adam', loss='sparse_categorical_crossentropy', and metrics=['accuracy']. We chose the 'sparse_categorical_crossentropy' function because our dataset includes integer labels and it works for most multi-classification tasks.  

14. Trained both submodels with identical EarlyStopping and ModelCheckpoint callbacks.  
  
  
## Model Ensembling  
  
Ensembling models entails combining the individual predictions of multiple models on the same dataset, in an attempt to make better predictions on that dataset. Ensemble models can improve upon the predictive performance of their submodels. If different models make different types of errors, we may be able to reduce the overall error rate by combining their predictions. 

In this project, we combined our two submodels' predictions in ensemble_model, which averaged the submodels' output. Here, each model contributing to ensemble_model was weighted equally. It is possible to configure a weighted average ensemble in which better-performing submodels contribute more to the ensemble than poorer-performing submodels. 

There are additional techniques for combining submodel predictions. In bootstrap aggregating, multiple models are trained on different subsets of the same training data and then ensembled. Boosting models occurs when models are trained sequentially, allowing later models to correct the errors made by earlier models. The voting technique makes a final prediction by taking a majority vote of the predictions made by the various submodels. 

Ensemble models can yield improved accuracy over their individual submodels by reducing overfitting. They may exhibit more robustness to changes in input data than their submodels. On the other hand, ensemble models can entail increased complexity, reduced ease of interpretability, and greater computational costs than their submodels individually.    
  
  
We took the following steps to prepare for and build the ensemble_model:

1. Defined the full file paths to our best saved first_model and second_model, and loaded them from saved. keras.
   
2. Extracted labels from the TensorFlow datasets (training_set, testing_set, validation_set) we had created by using the tf.keras.preprocessing.image_dataset_from_directory method. Ensemble models need labels to compute loss (by comparing predictions to true lables) and update models during training. 
  
3. Generated submodel predictions for the training and validation datasets with shape (None, 4):  
    * preds_first_model_train = first_model.predict(training_set)
    * preds_second_model_train = second_model.predict(training_set)
    * preds_first_model_val = first_model.predict(validation_set)
    * preds_second_model_val = second_model.predict(validation_set)
  
4. Defined EarlyStopping and ModelCheckpoint callbacks and a filepath to save the best ensemble_model. 
  
5. Built, compiled, and trained the ensemble_model, in the same manner as the two submodels, to process the combined predictions.   
    * Defined ensemble_input_train as the simple average of the first_model's predictions on the training_set and the second_model's predictions on the training_set.
    * Defined ensemble_input_val as the simple average of the first_model's predictions on the validation_set and the second_model's predictions on the validation set.
    * Defined ensemble_input as the input layer for the ensemble_model, with Input(shape=(4,)) because
        * The submodels' outputs were predictions of shape (None, 4), which ensemble_model takes as inputs.  
        * The ensemble_model does not take the image datasets fed to the submodels.  
    * Added a Dense layer as the output layer, final_output = Dense(4, activation='softmax')(ensemble_input).  
    * Defined ensemble_model as Model(inputs=ensemble_input, outputs = final_output).  
    * Compiled ensemble_model with optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'].  
    * To train the ensemble_model, we used ensemble_input_train as our x and y_train - the true labels from our training_set - as our true labels.  
    
      
## Transfer Learning   

An alternative to ensembling two models is chaining them. After pre-training, a model can be applied to a new, specific dataset and classification task, in a process called transfer learning. The pre-trained model's weights, optimized during pre-training, become the starting point for training on a new, often smaller, dataset. The model learns the specifics of the new task while leveraging the general features it learned during pre-training. In our project, the smaller dataset consisted of the CT-Scan images with different types of chest cancer versus normal cells. 
  
  
## Chaining Models  

Chaining two models means creating a composite model, where the first model's output becomes the input for the second model's layers. The output of the first model in a two-model chain is not a classification, but feature vectors that help the second model's layers make a classification. Chaining two models results in a single model that can be trained end-to-end. Model chaining can be performed using the Functional API in a Keras framework. The Functional API enables flexible connection of layers and models. 
  
We took the following into account when chaining model_one and model_two:  
   
1. Saved and renamed fist_model and second_model as mod_first_model and mod_second_model, because we needed to make modifications to the original code.
 
2. Removed the final, dense layer from mod_first_model because we didn't want it to generate a vector representing class probabilities. We used mod_first_model only as a feature extractor. We allowed mod_second_model to generate the class probabilities.  
    * We defined a new output layer for feature extraction, mod_first_model_output = x.  
    * The final output of mod_first_model became the result of the Dropout layer immediately preceding its final output layer.  
      
3. Because data augmentation and rescaling were necessary only in the first of two chained submodels, we removed augmentation and rescaling layers from mod_second_model.  
   
4. Removed the dropout, convolutional, and pooling layers present in second_model from mod_second_model because such layers were already present in the ResNet50 base of mod_first_model. Not removing these custom layers risked: 
      
   * Over-parameterization, which could have resulted in a model with too many parameters. Over-parameterized models  
      * Risk of overfitting, especially on small datasets.  
      * Require more computational resources.  
      * Make training unstable.    
  
   * Redundant feature extraction, which could have caused computational inefficiency and degradation of learned features, as custom layers "over-process" the features.    
  
   * Loss of transfer learning benefits, if custom layers on top of ResNet50 had disrupted the transfer learning process.  
      * If the transfer learning process gets disrupted, training begins again and we lose the benefit of pre-trained weights.  
      * If custom layers don't complement the ResNet50-extracted features, model effectiveness drops.    
      * Too many layers could undermine the pre-trained model's ability to generalize.    

   * Training instability, which can occur when excessive layers make the model architecture deeper and more complex than necessary.  
  
   * Overfitting, when the model memorizes the training data instead of learning generalizable patterns.    
  
5. Removed second_model's Flatten layer from mod_second_model, as it was no longer necessary.   
  
6. Defined one Dense and one Dropout layer before defining an output layer capable of producing a four-class classification.  
    
7. Defined but did not compile and train mod_first_model and mod_second_model, instead training them as one chained model, chained_model.  

8. Chained mod_first_model and mod_second_model by:  
   
   * Defining variable 'mod_first_model_output' to hold the feature vector output from 'mod_first_model'; mod_first_model_output = mod_first_model.output.    
   * Passing feature vector mod_first_model_output into mod_second_model, which would take the feature vector and process it further through its layers.     
   * Defining variable 'mod_second_model_output' to hold mod_second_model's output (classification probabilities) by setting mod_second_model_output = mod_second_model(mod_first_model_output).       
   * Defining a new Keras model called chained_model that chains together mod_first_model and mod_second_model into one model by  
       * Setting chained_model = Model(inputs=mod_first_model.input, outputs=mod_second_model_output),
       * Where inputs=mod_first_model.input specified that the input to chained_model is the same as the input to mod_first_model, and  
       * Where outputs=mod_second_model_output specified that chained_model's output is taken from mod_second_model_output,  
       * Which was the output of mod_second_model after the feature vector was passed from mod_first_model.  
  
9. Defined mod_first_model_output as mod_first_model_model.output, making mod_first_model's layers the first 'link' in the chain.  
   
10. Defined mod_second_model_output as mod_second_model(mod_first_model_output), to pass the first 'link's' output to the second 'link' in the chain, mod_second_model.      
11. Defined the output of the second link in the chain as mod_second_model_output. This allowed us to define the composite model, chained_model, as Model(inputs=mod_first_model.input, outputs=mod_second_model_output).  
    
12. Specified optimizer = Adam(), defined a filepath to save chained_model's best model, and defined equivalent EarlyStopping and ModelCheckpoint callbacks as used previously.  
    
13. Trained chained_model on the dataset training_set and set validation_data as the validation_set.   
  
  
## Terms Describing Model Performance  
  
The following metrics, when used together, provide a comprehensive evaluation of a model’s performance. That is, they communicate how accurate, consistent, and well-suited for generalization any given model is for a task.  
  
### Model Accuracy  
  
Model accuracy is calculated as the number of correctly classified predictions divided by the total number of predictions. It indicates how well the model performs overall, but doesn't consider any costs related to misclassifications or class imbalance. 
 
### Model Loss  
  
Model loss is a numerical value that represents the difference between predicted values and actual values. It can be calculated in various ways, using loss functions specific to classfication tasks (e.g., categorical_crossentropy, sparse_categorical_crossentropy). Loss plays an important role in a model's straining process, adjusting weights and minimizing errors. 
  
### Model Balance  
  
A model's Balance Score is calculated as 1 - Max Gap, where Max Gap is the largest absolute value difference between a model's training, validation, and testing accuracies: max_gap = max(abs(training accuracy - validation accuracy), abs(validation accuracy - testing accuracy), abs(training accuracy - testing accuracy)).   
The Balance Score measures consistency in a model's accuracy across training, validation, and testing. The more balanced the model, the more consistently it performs across datasets. Balanced models are more likely to perform well on new, unseen data than unbalanced models, making them more useful.  
  
When balance is good, the model's performance metrics (accuracy, loss) across training, validation, and testing datasets are both:  
   a. Similar, exhibiting small gaps between datasets, indicating reasonable generalization, and  
   b. Consistent, with validation and testing results not significantly better nor worse than training results.  
  
When balance is poor, the model overfits to the training data and performs significantly worse on unseen data. The Balance Score is intended to keep the model from overfitting (memorizing training data but failing on unseen data) and underfitting (failing to learn meaningful patterns from the data). Poorly balanced models are not robust, but very senstive to variations in data. A model with strong balance produces consistent and reliable results across datasets, making it effective and reliable.  
  
### Model Average Accuracy  
  
A model's Average Accuracy Score is the mean of the model's training accuracy, validation accuracy, and testing accuracy values. The Average Accuracy Score indicates how well a model performed across all three datasets, a measure of absolute performance. 
    
### Model Composite Score  
  
A model's Composite Score is a weighted combination of its Balance and Average Accuracy Scores: Composite Score = w1 × Balance Score + w2 × Average Accuracy, where w1 and w2 are weights assigend to the component scores, respectively. Because it measures consistency and absolute accuracy, it is regarded as an indication of a model's overall performance. Composite Score is a single metric that considers both the importance of model stability (balance) and the importance of overall performance (accuracy).   
  
  
## Evaluating All Four Models  
  
first_model and second_model (the submodels) needed to be evaluated on the unseen testing_set dataset to get unbiased performance metrics. Becaus first_model, second_model, and chained_model were already trained on the training_set and validated on the validation_set, we evaluated these three models on the testing_data.  
  
With the ensemble_model, evaluation was a matter of   
a) averaging the predictions from the two submodels on the unseen testing_set,  
b) extracting the labels from the testing_set, and   
c) estimating ensemble loss and ensemble accuracy by requesting ensemble_model.evaluate(ensemble_predictions, y_test).  
  
  
## Tables Of Results  
  
| Model          |   Train Loss |   Train Accuracy |   Validation Loss |   Validation Accuracy |   Test Loss |   Test Accuracy |    
|:---------------|-------------:|-----------------:|------------------:|----------------------:|------------:|----------------:|  
| first_model    |     0.437609 |         0.822186 |          1.02791  |              0.652778 |    1.17075  |        0.514286 |  
| second_model   |     0.572898 |         0.743883 |          0.727373 |              0.763889 |    1.57829  |        0.469841 |  
| ensemble_model |     1.38543  |         0.257749 |          1.44298  |              0.222222 |    1.38834  |        0.234921 |  
| chained_model  |     0.538555 |         0.774878 |          0.95535  |              0.638889 |    0.956217 |        0.568254 |  
  
    
| Model          |   Balance Score |   Average Accuracy |   Composite Score |  
|:---------------|----------------:|-------------------:|------------------:|  
| first_model    |        0.6921   |           0.663083 |          0.677591 |  
| second_model   |        0.705952 |           0.659204 |          0.682578 |  
| ensemble_model |        0.964473 |           0.238297 |          0.601385 |  
| chained_model  |        0.793376 |           0.660674 |          0.727025 |  
  
  
## Model Performance Summaries  
  
first_model  
  * Training performance: With high accuracy and relatively low loss, first_model learned well on the training dataset. 
  * Validation performance: Moderate accuracy but higher loss suggested overfitting. first_model performed better on the training data than on the unseen data.  
  * Testing performance: A further drop in accuracy and increase in loss with the testing data confirmed that first_model generalized poorly to new data.  
  * Demonstrated moderate balance and consistency, but not the highest composite score of all four models.  
  * Conclusion: first_model is strong on training data but overfits, resulting in poor generalization.  
  
second_model  
  * Training performance: second_model demonstrated moderate training accuracy and loss, suggesting reasonable learning ability on seen data.   
  * Validation performance: accuracy and loss improved on the validation data, indicating less overfitting and better generalization than first_model.    
  * Testing performance: a significant drop in accuracy and increase in loss signaled overfitting and poor generalization on fully unseen data.  
  * Demonstrated moderate balance and average accuracy, which appear to be masking the poor testing performance. 
  * Conclusion: second_model generalized well to the validation data but could not maintain similar performance on the test data.  
  
ensemble_model  
  * Training performance: Very low accuracy and high loss indicated underfitting; ensemble_model failed to capture meaningful patterns in the training data.    
  * Validation performance: Similarly low accuracy, coupled with high loss, indicated ensemble_model did not improve performance over either of the submodels. 
  * Testing performance: Again, accuracy and validation scores confirm the chosen ensemble strategy did not improve performance. Giving the two submodels equal weight when averaging their predictions was ineffective at predicting class assignments. 
  * Demonstrated the highest balance score but suffered from the lowest accuracy, reducing its utility in practice.  
  * Conclusion: Very low training, validation, and test accuracy scores - coupled with high training, validation, and testing losses, indicated ensemble_model struggled to fit and generalize across all datasets. ensemble_model demonstrated the poorest optimization and learning capability, despite the high balance score.  
   
chained_model   
  * Training performance: Moderate accuracy and loss scores indicated a well-trained model with a good learning process and limited overfitting.  
  * Validation performance: lower accuracy than training accuracy, with more loss than training loss, suggested some overfitting.  
  * Testing performance: Because accuracy was higher, and loss lower, than with first_model and second_model, chained_model exhibited better generalization to unseen data than the submodels.  
  * Achieved the highest composite score among the four models, demonstrating reasonable accuracy and consistency.  
  * Conclusion: chained_model achieved the best generalization, balancing, training, validation, and testing performance among all four models.   
  
With decent learning on the training data, mild overfitting, and the best generalization of all four models, the chained_model demonstrated the best overall performance. It also achieved the highest composite score, indicating it was the most balanced and accurate of the models. Our recommendation is to use the chained_model as the preferred choice of models.  
  
  
## Potential Next Steps  
  
The accuracy results for ensemble_model were noticeably low. It is unusual for an ensemble model that combines its submodels' output to have lower accuracy than its submodels. Such results can indicate an issue with how predictions were averaged, if the models' outputs are raw logits or probabilities. Because the two classification models are generating averageable probabilities, however, averaging errors are not at play.  
  
Likewise, we can rule out the possibility that the model was trained using pseudo-lables rather than true labels, since we explicitly specified the relevant true labels as the validation_data. Pseudo-labeling would have occured if we trained the ensemble model on the submodel predictions as labels, instead of using the true labels.  
  
Finally, a problem with the evaluation methodology doesn't explain the lower accuracy scores for ensemble_model. The evaluation of the ensemble model was consistent with how the model was trained (e.g., on averaged predictions).   
  
It could be that the two submodels are underperforming or have biases. If true, averaging the two submodels' predictions would not necessarily improve performance. It's also possible that averaging the submodels' predictions exacerbated weaknesses in the two models if the models made similar errors.   
  
Overfitting or underfitting could also be a factor. Averaging predictions from overfitting models could result in poor generalization to unseen data. Averaging predictions could also be problematic if the submodels were underfitting, failing to capture complex patterns.   
  
It may be the case that simple averaging is not appropriate when ensembling first_model and second_model. Averaging predictions when one model submodel is significantly better than the other can dilute the effectiveness of the stronger model. In such situations, ensembling using weighted averaging instead of simple averaging could improve performance. A possible next step could be ensembling first_model and second_model with weighted averages.  

