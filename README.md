# PconsC4 - Distance Prediction

An extension to the original Pcons4 model. This model would be capable of predicting the distance between two residues using the UNet++ Architecture.

# Predictions

Follow the steps below to make your own prediction using the deeper U-Net model.
Run the following command in the src folder.

```python3
python3 predict.py alignment.a3m output
```

alignment.a3m correponds to the alignment file. 
output is the name of the output file which had the predictions from the U-Net model. 
