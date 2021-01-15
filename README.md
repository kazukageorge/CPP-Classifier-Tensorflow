# Clathrin Coated Pit Classifier 

This is repository has the codes used to classify the traces of auxilin and clathrin molecules thats cooperate to form a clathrin coated pits inolved in how cells eat and transport outside-the-cell material. Images were collected using a micrscope.

The goal is to minimize the time needed to classify which datasets are of interests - let the machine make the decision, and humans will check the dataset for the final decision. 

We have about 22k samples of datasets. We divide 60% to training, 20% to validation and 20% to testing. Accuracy of >90%. 

This work was collaborated with Jeremy Gygi. 

## Preliminary info
We are interested in how cells "eat" extracellular materials (ex. virus, protein, large molecules). One way is by a process called clathrin mediated endocytosis, where two key molecules, clathrin and auxilin, can be used to determine if the cargo was entered successfully. Multiple clathrin molecules gather to make a circular shape "Pit" and at the onset of completion, auxilin molecules signal the disassembly of the clathrin coated pit (CCP). This means that when the pit is forming, clathrin signals gather (increases), and near the peak, auxilin intensity will increase indicating the onset of uncoating.

<img src="/img/img2.png" width="800"/>

These molecules were captured using microscopes and were detected/tracked using MATLAB. The metadata is saved in a `.mat` format. 

<img src="/img/img1.png" width="500"/>

For more info about the biology, please read [Dynamics of Auxilin 1 and GAK in clathrin-mediated traffic](https://rupress.org/jcb/article/219/3/e201908142/133624/Dynamics-of-Auxilin-1-and-GAK-in-clathrin-mediated)

For more info about how molecules were tracked, please read [Advances in Analysis of Low Signal-to-Noise Images Link Dynamin and AP2 to the Functions of an Endocytic Checkpoint](https://www.cell.com/developmental-cell/fulltext/S1534-5807(13)00382-1?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1534580713003821%3Fshowall%3Dtrue)

## Code Flowchart
1. Load the traces for clathrin and auxlin and parse the metadata into a dataframe
2. Average out the traces data (vector) into a scalar value
3. Data processing. eg. Normalize the data for training using `sklearn.preprocessing.StandardScaler`
4. Split the data randomly into training, validation, testing set
5. Train the model using your choice of NN. I used a simple logistic regression with SGD/ADAM with MSE, which gives accuracy of ~93%


## Requirements
I am using macOS Mojave 10.14.6. Installed packages are:

1. tensorflow   2.4.0
2. numpy    1.19.2
3. pandas   1.1.5
4. scipy    1.5.2
5. iPython  7.19.0

## Config

You will need to create an envrionment that has tensorflow. Type `conda env create -n tensorflow`, and install the above packages. Once installed, type `conda activate tensorflow`. Locate to the directory that has the repository and type `python main.py -i <dir_to_InputData>`










  
