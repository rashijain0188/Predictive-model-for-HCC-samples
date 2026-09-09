\# Predictive-model-for-HCC-samples



\*\*Project Description\*\*



The project created a machine learning model for predictive modelling to screen candidate biomarkers and find utility for diagnosis in tissue and serum. It is able to classify hepatocellular carcinoma samples from normal to the tumour, based on RNA-seq expression values of protein-coding RNAs. This aided in biomarker discovery.





\*\*Model Details\*\*



Model type: XGBoost



Input: RNA expression data- Raw counts



Genes: GPR50, MYH1, MT2A, E2F8, MYH4, MTNR1B, GLP2R, GNAO1, EGF, BUB1B, MMP3, PLK1, MMP1, MT1E, MT1F, FOXM1, E2F1, MYH7, CCL20, MMP9, CDC20, GLP1R, ADRA1D, MT1G, ADRA1A, IGF1, MT1X, CDK1, MYH8, GNG4, MYH13





\*\*User Manual\*\*



\*\***Installation**\*\*





\*\***R Environment**\*\*



Download and install R from the official R Project website.

Install RStudio as an optional integrated development environment for running R scripts.



Open R/RStudio and install the required packages as per requirements.txt file:



source("R/install\_packages.R")



The R version and package versions used for the analyses should be documented in R/README.md or the repository environment files.



\*\***Python Environment**\*\*



The predictive modelling workflow was developed using Python v3.11.5. The analysis can be run using Spyder or another Python IDE.



Install Anaconda/Miniconda or another Python distribution.



Create a dedicated environment using Python 3.11.5:



conda create -n hcc\_ml python=3.11.5

conda activate hcc\_ml



Install the required Python packages:



pip install -r requirements.txt



Launch Spyder:



spyder



The requirements.txt file contains the fixed package versions used for model development and evaluation to facilitate reproducibility.



\*\*Repository Structure\*\*

Predictive-model-for-HCC-samples/

│

├── README.md

│

├── Data/

│   ├── Train\_Dataset.csv

│   ├── Test\_Dataset.csv

│   ├── External\_Liver\_Tissue\_Dataset.xlsx

│   └── Serum\_Exosomes\_Dataset.csv

│

├── Scripts/

│   ├── CombatSeq\_Preprocessing\_Training\_DEGs.R

│   ├── CombatSeq\_Preprocessing\_Test\_External.R

│   ├── Training\_ML\_Models.py

│   ├── Training\_DL\_Models.py

│   └── External\_Validation.py

│

├── Model\_and\_Scaler\_Files/

│   ├── XGBoost\_best\_model.pkl

│   └── scaler.pkl

│

└── Requirements.txt



\*\*Folder Description\*\*

Data/ – Training, test, external liver tissue, and serum exosome datasets used in the analysis.

Scripts/ – R and Python scripts for preprocessing, DEG identification, machine-learning/deep-learning model training, and external validation.

Model\_and\_Scaler\_Files/ – Final trained XGBoost model and corresponding scaler used for prediction.

Requirements.txt – Fixed Python and R package versions required to reproduce the computational workflow.

README.md – Installation instructions, workflow description, and instructions for reproducing the analyses.



To facilitate reproducibility, the repository contains the preprocessing, model-training, validation and prediction scripts, together with fixed software/package versions. 

