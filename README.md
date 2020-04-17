# Congestion Control analysis in mmW-ns3

This project contains ongoing work of congestion control analysis using the ns3-mmW module.

The project has two different parts:

* mmWave channel traces generation: to generate the traces we use the ns3-mmWave module using the 3GPP channel model
* Machine learning: it uses Scikit-learn Python library 

##  mmWave channel traces generation

The scenario comprises a single  user and base station with constant traffic and static positions. 

Un-compress and build the ns3 simulato from `ns3-mmwave_small.tar.gz`. 

To generate the traces run the script `scenario.py`. There you can define the data-rates and distances between the base station and user equipment.

As output, traces are generated within the `traces` folder.

##  Machine learning

Machine learning algorithms are implemented in:
* `classify.py` for SVM
* `clustering.py` for K-means 

------------------------------------------

contact: ldiez@tlmat.unican.es