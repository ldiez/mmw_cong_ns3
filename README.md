# Congestion Control analysis in mmW-ns3

This project contains ongoing work of congestion control analysis using the ns3-mmW module.

The project has two different parts:

* mmWave channel traces generation: to generate the traces we use the ns3-mmWave module using the 3GPP channel model
* Machine learning: it uses Scikit-learn Python library 

##  mmWave channel traces generation
To run the scenario follow the next steps: 

* Un-compress and build the ns3 simulato from `ns3-mmwave_small.tar.gz`. 

```
tar -xzvf ns3-mmwave_small.tar.gz ns3-mmwave_small
```

The folder `ns3-mmwave_small` contains a simplified version of the [ns3-mmWave](https://github.com/nyuwireless-unipd/ns3-mmwave) simulator where no necessary modules have been removed. Also, pythonf bindings, eaxamples and tests are removed.

* Install basic ns3 dependencies as decribed in this [wiki](https://www.nsnam.org/wiki/Installation). The following should suuffice

```
apt-get install gcc g++ python python3
```

* To compile the simulator enter the folder
```
cd ns3-mmwave_small
```
* Configure it. The following command runs the configuration disables warnings trated as errors and enables c++14 standard. 
```
CXXFLAGS="-Wall --std=c++1y" ./waf configure
```
* Compile it.
```
./waf
```

The following modules should be compiled:
```
Modules built:
antenna (no Python)       applications (no Python)  bridge (no Python)        
buildings (no Python)     config-store (no Python)  core (no Python)          
internet (no Python)      lte (no Python)           mmwave (no Python)        
mobility (no Python)      mpi (no Python)           network (no Python)       
point-to-point (no Python) propagation (no Python)   spectrum (no Python)      
stats (no Python)         traffic-control (no Python) virtual-net-device (no Python)
```
* To generate the traces run the script `scenario.py`. There you can define:

  * Data rates in Mbps with the array `bws` 
  * Distances in meters between the base station and user equipment with the array `dists`.
  * Simulation time in ms with `times`

As output, traces are generated within the `traces` folder.

### Simulation traces

There are two types of traces:

* Traces from the physical. The file name follows the pattern `DlRxPhy_<bw>_<dist>.txt`. The columns represent:
    * Column 1: 'time' in seconds with high resolution
    * Column 2:  'tbSize' bytes sent to the physical channel
    * Column 3: 'mcs' index of the modulation and coding scheme (it depends on the channel and scheduler
    * Column 4: 'SINR in dB
* Traces from upper layers. The file name follows the pattern `static_<bw>_<dist.txt`. The columns represent:
  * Column 1: packet index
  * Column 2: packet arrival time in seconds
  * Column 3: RLC buffer occupancy at the time of the packet arrival
  * Column 5: Packet delay in milliseconds
  * Column 5: packet inter-arrival-time (IaT) in milliseconds

##  Machine learning

Machine learning algorithms are implemented in:
* `classify.py` for SVM
* `clustering.py` for K-means 

Both consume the traces from the upper layers (`static_<bw>_<dist>.txt`). The configuration parameters of both scripts can be modified with the following varaibles within them:

```
bws = [10, 20, 30, 40, 50, 100, 200, 500, 1000]
dists = [30]  # [10, 20, 30, 40, 50]
qth = 5e5
inFolder = './traces_default/'
outFolder = './cluster_out/'
```
These scripts are just a direct application of the SVM and k-means algorithms provided by the `` module
These scripts are just a direct application of the SVM and k-means algorithms provided by the [sklearn]() module

------------------------------------------

contact: ldiez@tlmat.unican.es