# PconsC4 - Distance Prediction

An extension to the original PconsC4 model. This model is be capable of predicting the distance between two residues using the modified UNet++ Architecture.
The input for the model is a multiple sequence alignment file and the output would be a matrix with the predictions.

<img src="/src/images/MSA.jpg" width="50" height="50">
![Input Multiple Sequence Alignemnt](/src/images/MSA.png){:height="50%" width="50%"} 
![Output Heatmap](/src/images/2.png){:height="50%" width="50%"}

# Predictions

Follow the steps below to make your own prediction using the deeper U-Net model.
Run the following command in the src folder.

```python3
python3 predict.py alignment.a3m output
```

alignment.a3m correponds to the alignment file. 
output is the name of the output file which had the predictions from the U-Net model. 
