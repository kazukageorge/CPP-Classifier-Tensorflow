# Clathrin Coated Pits Classififer using traces of Auxilin and Clathrin usng Tensorflow

This is repository has the codes used to classify the traces of auxilin and clathrin molecules thats cooperate to form a clathrin coated pits inolved in how cells eat and transport outside-the-cell material. Images were collected using a micrscope.

The goal is to minimize the time needed to classify which datasets are of interests - let the machine make the decision, and humans will check the dataset for the final decision. 

We have about 22k samples of datasets. We divide 60% to training, 20% to validation and 20% to testing. Accuracy of >90%. 

This work was collaborated with Jeremy Gygi. 

## Preliminary info
We are using 3D image data obtained from a microscope that tracks two molecules of interest -- clathrin and auxilin. These molecules are invovled when cells "eat" extra-cellular material (ex. virus, protein, large molecules). At the location where cell would like to invaginate the "food", bunch of clathrin molecules cluster right underneath the cell surface, and forms a "pit" (looks like a cage to ferry the food inside, ~70nm in diameter). When cage surrounds the food and the disassembly of each clathrin is decided by the arrival of auxilin. Hence, if auxilin shows up, the cell was successful to form a pit, and eat its food. 

For more info about the biology, please read [Dynamics of Auxilin 1 and GAK in clathrin-mediated traffic](https://rupress.org/jcb/article/219/3/e201908142/133624/Dynamics-of-Auxilin-1-and-GAK-in-clathrin-mediated)

For more info about how molecules were tracked, please read [Advances in Analysis of Low Signal-to-Noise Images Link Dynamin and AP2 to the Functions of an Endocytic Checkpoint](https://www.cell.com/developmental-cell/fulltext/S1534-5807(13)00382-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1534580713003821%3Fshowall%3Dtrue)

## Flowchart
Clathrin and Auxilin molecule images were acquired in `tif` format. Molecules were tracked and its metadata (intensity, number of frames, etc) were analyzed using MATLAB. 

![img1](/img/img1.png =100x)


## Requirements
* Tensorflow 

## Config

You will need to create an envrionment that has tensorflow. We are 

1. Open "Terminal"
2. type: source ~/tensorflow/venv/bin/activate
3. type: python masterscript.py












  
