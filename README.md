# Deep_learning_tutorial
This is a PhD level course on Deep Learning. For tutorials, we learn practical skills and discuss relevant details. This github repository contains slide decks and tutorial materials for anyone who would like to learn more about DL. Corresponding PPT slides can be found here: https://www.overleaf.com/read/wvhbvgbpkqnz

## Table of contents
* [Tutorial2](#Tutorial2)
* [Tutorial3](#Tutorial3)
* [Tutorial4](#Tutorial4)

## Tutorial 2.
For tutorial 2, we discuss H2o Flow. A basic introduction to H2o flow is covered in the following file: tutorial2 (1).Flow.
In order to run H2o Flow, you need to have the followings on your computer. 1) H2o application 2) Java
You can download H2o from [here] (https://www.h2o.ai/resources/download/)
You can download Java from [here] (https://www.java.com/en/)

Alternatively, you can run the following commands on your terminal
```
$ curl -o h2o.zip http://download.h2o.ai/versions/h2o-3.30.0.6.zip
$ cd ~/Downloads
$ unzip h2o-3.30.0.6.zip
$ cd h2o-3.30.0.6
$ java -jar h2o.jar
```
Finally, please click here to open Flow: http://localhost:54321/

Note: that either way you need to have java. 

Then, you download the tutorial and import it when you open H2oFlow Notebook. 

## Tutorial 3. 
This tutorial shows how a H2o deep learning model can be used to do supervised classification and regression. We will use R.  H2O's Deep Learning algorithm is a multilayer feed-forward artificial neural network. It can also be used to train an autoencoder. In this example we will train a standard supervised prediction model.

### How to use H2o in R?
There are some useful links that you can read: 
[@H2o R references] (https://docs.h2o.ai/h2o/latest-stable/h2o-r/docs/reference/index.html)
[@H2o comprehensive tutorial] (https://docs.h2o.ai/h2o-tutorials/latest-stable/H2OTutorialsBook.pdf)

H2o offers an R package that can be installed from CRAN, and a python package that can be installed from PyPI. Or, it can be downloaded from here: http://h2o.ai/download.

### set up
```

 library(h2o)
 h2o.init(nthreads = -1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O
 loan_csv <- "https://raw.githubusercontent.com/h2oai/app-consumer-loan/master/data/loan.csv"
 data <- h2o.importFile(loan_csv)
 dim(data) # [1] 163987     15

 data$bad_loan <- as.factor(data$bad_loan)
 h2o.levels(data$bad_loan)
 splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.7, 0.15),  #partition data into 70%, 15%, 15% chunks
                         seed = 1)  #setting a seed will guarantee reproducibility
 train <- splits[[1]]
 valid <- splits[[2]]
 test <- splits[[3]]

 y <- "bad_loan"
 x <- setdiff(names(data), c(y, "int_rate"))  #remove the interest rate column because it's correlated with the outcome
```


 ### Train a default DL
 First we will train a basic DL model with default parameters. The DL model will infer the response 
 distribution from the response encoding if it is not specified explicitly through the `distribution` 
 argument.  H2O's DL will not be reproducible if it is run on more than a single core, so in this example,  the performance metrics below may vary slightly from what you see on your machine.
 In H2O's DL, early stopping is enabled by default, so below, it will use the training set and 
 default stopping parameters to perform early stopping.

```
 dl_fit1 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit1",
                            seed = 1)
```

 Train a DL with new architecture and more epochs.
 Next we will increase the number of epochs by setting `epochs=20` (the default is 10).  
 Increasing the number of epochs in a deep neural net may increase performance of the model, however, 
 you have to be careful not to overfit your model to your training data.  To automatically find the optimal number of epochs, you must use H2O's early stopping functionality (`stopping_rounds`).  Unlike the rest of the H2O algorithms, H2O's DL will use early stopping by default, so for comparison we will first turn off early stopping.  We do this in the next example  by setting `stopping_rounds=0`.

`epochs`: Specify the number of times to iterate the dataset. The value can be a fraction. This option defaults to 10. 

```
 dl_fit2 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit2",
                            #validation_frame = valid,  #only used if stopping_rounds > 0
                            epochs = 20,
                            hidden= c(10,10),
                            stopping_rounds = 0,  # disable early stopping
                            seed = 1)
```

 Train a DL with early stopping
 This example will use the same model parameters as `dl_fit2`. This time, we will turn on 
 early stopping and specify the stopping criterion.  We will also pass a validation set, as is
 recommended for early stopping.
 `score_interval`: Specify the shortest time interval (in seconds) to wait between model scoring. This option defaults to 5.
 
 `stopping_rounds`: Stops training when the option selected for `stopping_metric` does not improve for the specified number of training rounds, based on a simple moving average. To disable this feature, specify `0`. The metric is computed on the validation data (if provided); otherwise; training data is used. The deafult is 5.
 
 `stopping_metric`: Specify the metric to use for early stopping. Options are elaborated here: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/stopping_metric.html
 
 `stopping_tolerence`: Specify the relative tolerance fir tge netric-based stopping to stop training if the improvement is less than this value. This option defaults to 0. These stopping options are used to increase performance by restricting the number of models that get built. 
 https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/stopping_tolerance.html

```
 dl_fit3 <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit3",
                            validation_frame = valid,  #in DL, early stopping is on by default
                            epochs = 20,
                            hidden = c(10,10),
                            score_interval = 1,           #used for early stopping
                            stopping_rounds = 3,          #used for early stopping
                            stopping_metric = "AUC",      #used for early stopping
                            stopping_tolerance = 0.0005,  #used for early stopping
                            seed = 1)
```

Let's compare the performance of the three DL models

```
 dl_perf1 <- h2o.performance(model = dl_fit1,
                            newdata = test)
 dl_perf2 <- h2o.performance(model = dl_fit2,
                            newdata = test)
 dl_perf3 <- h2o.performance(model = dl_fit3,
                            newdata = test)
```

Print model performance
```
 dl_perf1
 dl_perf2
 dl_perf3
```
Retreive test set AUC

```
 h2o.auc(dl_perf1)  # 0.6774335
 h2o.auc(dl_perf2)  # 0.678446
 h2o.auc(dl_perf3)  # 0.6770498
```
Scoring history
```
 h2o.scoreHistory(dl_fit3)
```
Look at scoring history for third DL model
```
 plot(dl_fit3, 
     timestep = "epochs", 
     metric = "AUC")
```
Now let's go look at tutrial3_2 document. Pdf file (tuorial2_2, sorry for the confusing filename) is knitted version of tutorial3_2.Rmd. If you are interested you can run the R markdown file. If your computer does not have enough memory/space, please read the Pdf. 

## Sources
This is inspired by Michal Kurka [@michalkurka](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.R)

## Tutorial 4.
In this tutorial, we would like to explore tensorflow in python.
Why should we use tensorFlow?
TnesorFlow makes it easy for beginners and experts to create machine learning models for desktop, mobile, web and cloud. It provides a collection of workflows to develop and train models using Python or JavaScript, and to easily deploy in the cloud, on-prem, in the browser, or on-device no matter what lanaguage you use.

### resources
Firstly, here are some resources on TensorFlow: https://www.tensorflow.org/overview
Guide to TensorFlow: https://www.tensorflow.org/guide

### installation
To install python, please look at this doc. (https://www.tensorflow.org/install)
There are three ways to install TensorFlow.
* Download a package. (https://www.tensorflow.org/install/pip)
```
# Requires the latest pip
$ pip install --upgrade pip

#current stable release for CPU and GPU
$ pip install tensorflow
```
* Run a TensorFlow container (https://www.tensorflow.org/install/docker)
```
$ docker pull tensorflow/tensorflow:latest  # Download latest stable image
$ docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server 
```

* Google collab
This option does not require installation. 

### Basic model (source: https://www.tensorflow.org/tutorials/quickstart/beginner)

#### Set up TensorFlow
```
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

#### Load a dataset: MNIST
We will convert the sample data from integers to floating-point numbers:
```
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

#### Build a machine learning model
Build a tf.keras.Sequential model by stacking layers.
```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

We will discuss similar topics in tutorial 4. For more details, please go to tutorial4 files in the main branch.
