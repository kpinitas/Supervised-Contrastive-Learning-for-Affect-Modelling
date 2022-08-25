# Supervised Contrastive Learning for Affect Modelling

This is the code for the Supervised Contrastive Learning for Affect Modelling paper.

The code is written in Python and requires the Anaconda platform. 

## Environment

Create a new environment using the scl_env.yml image
```bash
conda env create -f scl_env.yml
```
## Dataset

You should also download the RECOLA Database from: https://diuf.unifr.ch/main/diva/recola/download.html and create a csv file that contains features and affect values for each particpant with name P[participant_id].csv e.g. P16.csv

## Scripts

From the files you will download run ``` main.py ```

## Important Note
The csv files should be saved inside a folder with name "PROCESSED-DATA" and place it at the same directory as the code
