# Predictive-model-for-HCC-samples

**Project Description**

The project created a machine learning model for predictive modelling to screen candidate biomarkers and find utility for diagnosis in tissue and serum. It is able to classify hepatocellular carcinoma samples from normal to the tumour, based on RNA-seq expression values of protein-coding RNAs. This aided in biomarker discovery.


**Model Details**

Model type: XGBoost

Input: RNA expression data- Raw counts

Genes: GPR50, MYH1, MT2A, E2F8, MYH4, MTNR1B, GLP2R, GNAO1, EGF, BUB1B, MMP3, PLK1, MMP1, MT1E, MT1F, FOXM1, E2F1, MYH7, CCL20, MMP9, CDC20, GLP1R, ADRA1D, MT1G, ADRA1A, IGF1, MT1X, CDK1, MYH8, GNG4, MYH13


**User Manual**

This user manual contains the steps required to load and run the project in Python v3.11.5.

Installing Spyder Any Python IDE can be used. Here the operation with Spyder is explained. Spyder IDE can be used online with https://mybinder.org/, installed using external distributor, e.g. Anaconda or stand-alone in your system. Refer to https://docs.spyder-ide.org/current/installation.html to install the latest version of Spyder as per your system. Create a new environment in conda as depicted in the HTML page.

It is always recommended to set a directory and download all files there for ease of use. This can be done by:

pip install os

import os

os.chdir(path)


In the last command, in place of path, you can add the path of your directory. OR load into an existing folder directly as done in the Predicting of samples.py.

After this, load the necessary libraries to run the model. These include:

NumPy: 2.2.6

Pandas: 2.3.2

Matplotlib: 3.10.5

Joblib: 1.5.2

pickle: uses Python standard library

Tkinter: 8.6

os: uses Python standard library


The commands for installing these are:

pip install numpy==1.26.4

pip install pandas==2.2.0

pip install matplotlib==3.8.2

pip install joblib==1.3.2


As you type these commands one by one, press Enter and it will start installing. It might take some time but once it is done, you are ready to run the model.

Download the .pkl model in the chosen directory.

Submit the raw counts present in data file in CSV (comma-separated) format as input when asked while running the model.

Open the script file in Spyder and click on run file to run the program. You will be asked to upload the data once the code starts running. Upload the downloaded data file and let the code run to completion.

An output file results folder as predictions.csv will automatically be downloaded in your directory. It can be opened in MS Excel or Google Sheets and the classification results can be interpreted for each Patient ID either as Normal or Tumor. 
