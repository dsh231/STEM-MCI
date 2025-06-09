# STEM-MCI
This repository includes a Jupyter Notebook (Run_Models.ipynb) that implements and runs various machine learning models for predicting cognitive status based on time-series ADL (Activities of Daily Living) data.

***File Overview***

Run_Models.ipynb: Main notebook to execute the end-to-end ML pipeline. It includes:

Data loading and preprocessing

Model training (e.g., LSTM, Random Forest, etc.)

Performance evaluation (accuracy, F1-score, confusion matrix, etc.)

/data/ (optional): Folder to store raw .pickle, .csv, or processed input data.

***Requirements***

Make sure you have the following installed:

Python 3.8+

tensorflow ≥ 2.9

pandas, numpy, scikit-learn, matplotlib, seaborn

(Optional) tqdm, pickle, etc.

You can install the requirements via pip:

```cmd
pip install -r requirements.txt
Or manually:
```
```cmd
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn tqdm
```

***How to Run***

Launch Jupyter Notebook:
```cmd
jupyter notebook
```
Open Run_Models.ipynb and follow the cells step by step:

Modify the file paths if needed

Ensure input data (e.g., .pickle) is in the correct location

Configure any hyperparameters or model settings as needed

Execute all cells for training and evaluation
