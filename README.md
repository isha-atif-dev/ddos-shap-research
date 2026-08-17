The data was loaded, cleaned and divided into six time periods representing different stages of a real deployment. The model was trained on the earliest time period only, which contained just three attack types, and achieved a baseline F1 of 0.8125. Each later time period contains completely different attack types, which means the model will encounter traffic patterns it has never seen before, setting the stage for explanation drift to occur.


 File 1, Data LoaderFile 1, Data Loader

350,000 rows, 7 attack types, 88 columns
BENIGN traffic is only 2,122 rows, less than 1%, class imbalance noted

File 2, Preprocessing

Data covers one single day, 3 November 2018, 9am to 5pm
Different attack types appear at different times of the day, genuine temporal variation
30,710 missing and infinity values found and fixed using median
SimillarHTTP column dropped due to mixed data types

File 3, Baseline Training

Batch 1 contains only Portmap, NetBIOS and BENIGN, the other attack types come in later batches
Baseline model scored F1 of 0.8125 and AUC-ROC of 0.7735, these are the reference numbers
Top feature by SHAP was Fwd Packet Length Min, followed by Source Port and Destination Port
Each batch has a different mix of attacks, so SHAP rankings will almost certainly shift over time, which is exactly the explanation drift this research is designed to detect