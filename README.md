# Molecule Graph Generation
This repository aims at providing minimal implementation of Graph Convolutional Networks (GCNs) based Variational Graph AutoEncoders (VGAE) for molecule generation. VGAE is being used here to generate new molecular graphs that have similar statistical distribution as that of the learned distribution of the training dataset.
## Usage
### Data
***ZINC*** dataset from PyTorch Geometric (PyG) dataset collection is being used here. It contains 12k molecular graphs with 38 heavy atoms. Moreover, for each molecular graph, the node features are the types of heavy atoms and the edge features are the types of bonds between them. Besides each node in each of the molecular graphs is represents an atom and the edge (link) between any two atoms represents the checmical bonding between them.
### Training, Validation, and Testing
The current implementation provides three imperative functions:-
- To train a new network : `train()`
- To validate the network being trained : `val()`
- To test a preTrained network : `test()`
- The average loss, area under curve score, and associated average precision for the trained model are printed after every epoch.
- All hyperparamters to control training, testing, and generation of the molecular graphs are provided in the given `.py` file.
## Generated *ZINC* molecule graphs
`gen_graphs` variable is set to 5 in the current implementation to generate 5 new ***ZINC*** molecule graphs, however, it can be set to any number based on the requirements.
![alt text](https://github.com/fork123aniket/Molecule-Graph-Generation/blob/main/Images/1.PNG)
![alt text](https://github.com/fork123aniket/Molecule-Graph-Generation/blob/main/Images/2.PNG)
![alt text](https://github.com/fork123aniket/Molecule-Graph-Generation/blob/main/Images/3.PNG)
![alt text](https://github.com/fork123aniket/Molecule-Graph-Generation/blob/main/Images/4.PNG)
![alt text](https://github.com/fork123aniket/Molecule-Graph-Generation/blob/main/Images/5.PNG)
