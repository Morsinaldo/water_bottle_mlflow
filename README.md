# Water Bottle Classification with MLFlow

## :rocket: Introduction

Notebooks are very popular in data science and machine learning, especially when we are performing exploratory data analysis and when we want to check the output of each cell of code. However, if we want to go beyond automating our processes using MLOps best practices, it is ideal to create a pipeline using scripts and not notebooks. Tools such as MLFlow and Hydra assist in this automation process, and with this in mind, this project aims to show a simple example of using these tools.

<p align='center'>
    <img src='./images/model_card.png'>
    Figure 1: Model card.
</p>

For more details about MLFlow, specifically, you can read my article on Medium on this link: . In this repository you will find details on how to reproduce this project and get the same results.

## :mag_right: The problem

The dataset used for training the model can be found on [Kaggle](https://www.kaggle.com/datasets/chethuhn/water-bottle-dataset) and contains 486 images divided into three classes:

1. Full Water Level (308 images)
2. Half Water Level (139 images)
3. Overflowing (39 images)

Figure 2 shows some images from the dataset.

<p align='center'>
    <img src='./images/dataset.png'>
    Figure 2: Images from dataset.
</p>

So, this project aims to be more of a didactic example of MLFlow usage than a solution to a specific problem in the real world, but if we use our imagination, we can find some applications. For example, imagine a production line of a company that sells bottles of soda, water, alcoholic beverages, etc., and this company wants to put hardware with a model implemented that classifies each bottle that leaves the production line and, at the end of a certain period, generates a report with the classifications obtained. This would be very useful for this company, given the thousands of bottles the machine produces per day.

## :computer: How to execute

To execute the MLFlow pipeline, you can follow the steps below:

__1__ - Create an account in [Weights and Biases](http://wandb.ai/) (WandB, for short).

__2__ - Clone this repo with the following command:
```
git clone https://github.com/Morsinaldo/water_bottle_mlflow.git
```

__3__ - Create the conda environment from [conda.yml](./conda.yml) file in `/water_bottle_mlflow/` with the following command.
```
conda env create -f conda.yml
```

It is from this virtual environment that the pipeline steps shown in figure 3 will be executed, and which is divided into the five components listed below.

<p align='center'>
    <img src='./images/pipeline.png'>
    Figure 3: Pipeline.
</p>

1. Fetch Data
2. Preprocessing
3. Data Segregation
4. Train
5. Test

Each component has a `conda.yml` file with the libraries used and their respective versions, thus creating fully separable and reproducible modules. Furthermore, all generated artifacts are uploaded to your WandB account, thus allowing the whole process to be trackable. The parameters that are passed to each step are in the [config.yaml](./config.yaml) file and you can change them to make it run :D. So let's take a look at each component separately.

### Fetch Data

As mentioned before, the dataset was downloaded from Kaggle and the images are divided into sub-folders named by their respective class. Then the images were zipped, put into Google Drive, and shared via the link in the script. 

__4__ - Execute the `fetch_data` step with the following command: 
```
mlflow run . -P hydra_options="main.execute_steps='fetch_data'"
```

__Note__: Run mlflow commands in the project's root folder.

This command will create the configured environment for the data download and will download the files from Google Drive through the URL indicated in the command. You can change this script for example to download data from AWS, Google Cloud, and Firebase, among others. The artifact generated by this component is called `raw_data`, as shown in figure 3, and will be used as input for the next component.

### Preprocessing

In this step we use classes found in the [images_processing.py](./preprocessing/helpers/images_processing.py) file: `SimpleProcessor` and `SimpleDatasetLoader`. The first class is responsible for resizing the images to the size passed as an initialization parameter of the object. The second class is responsible for loading the images from their folders, applying the preprocessors, and returning the dataset's features and labels. In this case only one processing was used to resize the images to (224,224), but you can create other classes in your project and pass the respective objects in a list format as parameters to the `SimpleDatasetLoader` class.

__5__ - Execute the `preprocessing` step with the following command:
```
mlflow run . -P hydra_options="main.execute_steps='preprocessing'"
```
__Note__: Run mlflow commands in the project's root folder.

The output of this component generates two artifacts: one with preprocessed features (`clean_data_features`) and another with labels (`clean_data_labels`).

### Data Segregation

This step is responsible for dividing the images that will be used for training, validation and test. The images were divided as follows: 75% for training and 25% for testing. From the 75% for training, 25% were separated for validation, thus obtaining the numbers shown in figure 4.

<p align='center'>
    <img src='./images/data_segregation.png'>
    Figure 4: Data Segregation.
</p>

__6__  - Execute the `data_segregation` step with the following command:
```
mlflow run . -P hydra_options="main.execute_steps='data_segregation'"
```
__Note__: Run mlflow commands in the project's root folder.

As a result, the output of this component generates six artifacts: `train_x`, `train_y`, `val_x`, `val_y`, `test_x` and `test_y`.

### Train

For training, a pre-trained Pytorch model called [VIT_L_32](https://pytorch.org/vision/main/models/generated/torchvision.models.vit_l_32.html) was used, which was encapsulated in a class called `MyVisionTransformerModel`, which can be found in the [model.py](./train/helpers/model.py) file. Object-orientation makes it easier (and more elegant) to implement training and prediction methods for the model. Also, in the `trian/helpers/` folder you can find another file called [images_processing.py](./train/helpers/images_processing.py), which contains a class called `ImageDataset`, responsible for creating the dataset in the format that will be loaded into Pytorch. I have kept this separate from the preprocessing stage because if you are going to work with TensorFlow or Keras, you will not need this class. So feel free to adapt these decisions as best you can in your project.

__7__ - Execute the `train` step with the following command:
```
mlflow run . -P hydra_options="main.execute_steps='train'"
```
__Note__: Run mlflow commands in the project's root folder.

The output of this component generates the fine tuned model for our project's task and the encoder of the target variable. In this case I used only the `LabelEncoder`, but if you look at the logs of this step you will notice that it is giving a higher weight to the Overflowing class. I encourage you to try using a different encoder than mine to try to improve the model results and also to use a different pre-trained model. Finally, over 50 epochs, the following result was obtained:

### Test

This step serves only to evaluate the performance of the model on the testbed and thereby decide whether or not it should be put into production. Therefore, this step has no output artifacts, only the confusion matrix and the metrics obtained in the test base. 

__8__ - Execute the `test` step with the following command:
```
mlflow run . -P hydra_options="main.execute_steps='test'"
```
__Note__: Run mlflow commands in the project's root folder.

### Executing Multiple Steps

If you want to execute two or more steps with just one command, you can pass a list separated by a comma, like the command below:
```
mlflow run . -P hydra_options="main.execute_steps='train,test'"
```

To execute all steps in pipeline

```
mlflow run . -P hydra_options="main.execute_steps='fetch_data,preprocessing,data_segregation,train,test'"
```

## :chart_with_upwards_trend: Model Performance
Figures 5 and 6 show, respectively, the confusion matrix and the plot of accuracy and error over the training stage epochs. With this, it can be seen that the model predicted too much of the Full water bottle class when it was actually Half water bottle. A possible future improvement would be to balance the number of class images with the increased data and try using other machine learning models. 

<p align='center'>
    <img src='./train/confusion_matrix.png'>
    Figure 5: Training Confusion Matrix.
</p>

<p align='center'>
    <img src='./train/loss.png'>
    Figure 6: Training Loss and Accuracy.
</p>

Training Metrics [burning-chocolate-56](https://wandb.ai/morsinaldo/water_bottle_classifier/runs/bjjr9omd?workspace=user-morsinaldo):
- Train Accuracy: 0.76
- Train Loss: 0.4867
- Validation Accuracy: 0.7912
- Validation Loss: 0.4867

Figure 7 shows the confusion matrix of the test base, and further down you can check the metrics for accuracy, precision, recall, and F1-Score. The accuracy of the test base was about 77%, which is reasonable, but we can improve it a lot in future updates. 

<p align='center'>
    <img src='./test/confusion_matrix.png'>
    Figure 7: Test Confusion Matrix.
</p>

Test Metrics [attractive-quiver-60](https://wandb.ai/morsinaldo/water_bottle_classifier/runs/piycb6qc?workspace=user-morsinaldo): 
- Test Accuracy: 0.77049
- Test Precision: 0.79535
- Test Recall: 0.77049
- Test F1: 0.77936

## :earth_americas: Deploy using Streamlit
In addition to MLFlow, a small application was made using Streamlit to symbolize the model in production. That is, given that the model has been trained, how can I use it to make predictions?

<p align='center'>
    <img src='./images/app_workflow.png'>
    Figure 8: Streamlit application.
</p>

__9__ - Enter in [app](./app/) folder and create the `app_env` environment with the `conda.yml` file using the following command:
```
conda env create -f conda.yml
```

__10__ - To start the application, run the following command with the `app_env` environment enabled:

```
streamlit run ./app.py
```

This will open a new tab in your browser with the format shown in figure 9:

<p align='center'>
    <img src='./images/app_1.png'>
    Figure 9: Streamlit application.
</p>

To perform a prediction, upload an image and click on the `predict` button, as shown in figure 10. The application will download the model and encoder stored in Weights and Biases, make the prediction and display the result just below.

<p align='center'>
    <img src='./images/app_2.png'>
    Figure 10: Streamlit application with image prediction.
</p>

## :leaves: Environmental Considerations

Given the environmental concerns raised in recent years because of the emission of CO2 into the atmosphere, a library named codecarbon was used to measure the amount of energy spent and the amount of CO2 released for the training of a neural network. As neural networks require a large computational cost, mainly GPU, the library was used to measure how much CO2 was released into the atmosphere and how much energy was spent during training .

- Energy consumed: 0.04285 kWh
- CO² emission: 0.005274 kg

This material is based on lectures by Professor Ivanovitch Silva [1] and the book Practical MLOps [2].
## :books: Reference

[1] Ivanovitch Silva Github. Available at: https://github.com/ivanovitchm/mlops - Accessed February 12, 2023.

[2] Practical MLOps. Available at: https://www.oreilly.com/library/view/practical-mlops/9781098103002/ - Accessed February 12, 2023.