# PconsC4 - Distance Prediction

An extension to the original PconsC4 model. This model is be capable of predicting the distance between two residues using the modified UNet++ Architecture.
The input for the model is a multiple sequence alignment file and the output would be a matrix with the predictions.

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="/src/images/MSA.png" width="250" height="250" title="Input MSA"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  <img src="/src/images/2.png" width="250" height="250" title="Output Heatmap"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;

The multiple sequence alignments were generated using JackHMMer and the features that were extracted the MSA are:

1. Gaussian Direct Coupling Analysis
2. APC-Corrected Mutual Information
3. Normalized APC-Corrected Mutual Information
4. Cross Entropy
5. Sequence Features

Using these 5 features, multiple fully convolutional neural network architectures were developed:

1. FC-DenseNet 103
2. U-Net
3. Recreation of the trRosetta Model
4. VGG 19
5. ResNet 50

Multiple different loss functions were tested on the various models. The loss functions are:

1. Focal Loss
2. Dice Loss
3. Categorical Cross Entropy
4. Mean Squared Error
5. Weighted CCE
6. Tversky Loss

# Predictions

Follow the steps below to make your own prediction using the deeper U-Net model.
Run the following command in the src folder.

```python3
python3 predict.py alignment.a3m output
```

alignment.a3m correponds to the alignment file. 
output is the name of the output file which had the predictions from the U-Net model. 
