# MOBAL

Mobility traces analysis: Studying mobility patterns, allowing insight into the operation of a smart city.

The main two programs that make up the steps described in the approach is the file `cf_tree.py` and `stop_point_detection.py` . These two programs alone can be used to replicate the findings in the thesis. The other programs are helpers that transform and store the data into a format that I have chosen to work with. As specified in the thesis, it is up to the user to decide how they want store the data. As such, the `fileio.py` program only works in this case, as it assumes that stay points are stored in separate tables per day. Additionally, the implementation found in this repository is not implemented in parallel or in an online fashion, here I simply save each step to a HDF5 file. All that needs to be done to run the approach in an online fashion, is to create wrappers for the `cf_tree.py` and `stop_point_detection.py` programs. The structure of both programs allows them to be used as such.

## **cf_tree.py**

This file is contains the Cluster Feature Tree implementation, within it are instructions and examples of how it is used. It corresponds to the third step in the Approach section.

## **stop_point_detection.py**

This file implements both the first and second step described in the Approach section of the thesis.

## Usage

To run each step in the pipeline you must use the `fileio.py` program. Running `./fileio.py --help` returns basic instructions on how to use each function provided. In order to produce the results documented in the Thesis document (cannot guarantee that they will be exact), one must run the following commands in the order specified.

### Prerequisite
Download the data-set from the following site:
https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/

### 1
Run the initial convert. This converts each CSV file found in the specified directory into a new HDF5 store.

```shell
./fileio.py --convert release/taxi_log_2008_by_id 'taxi_store.h5'
```
### 2
Calculate stay points found in the taxi trajectories within the specified range, with respect to a distance and time threshold. The following example calculates the stay points for every single taxi in the data-set. Note that this will take a long time, if you are running on a laptop it might be best to first take a smaller sample.

```shell
./fileio.py --spoints 1 10357 50 3 'taxi_store.h5'
```

### 3
Once you've calculate the stay points you wish to cluster, we need to split them up by day.

```shell
./fileio.py --sp_day_split 'taxi_store.h5'
```

### 4
Now the final step is to cluster the stay points. Here we have two options, 1) using the automatic threshold and 2) manually inputting a threshold.

**Automatic**
```shell
./fileio.py --auto_cluster DAY ORDER 'taxi_store.h5'
```

**Manual**
```shell
./fileio.py --cluster DAY ORDER THRESHOLD 'taxi_store.h5'
```
For best results: \
`DAY := (2,8)` \
`ORDER := 10` \
`THRESHOLD := 0.01`

### Retrieving results

Once clustering has been run once, it does not need to be run again.
To plot the clusters on the map, you can run the plot function. It simply reads the clusters calculated previously from the HDF5 store.

```shell
./fileio.py --plot_clusters DAY ORDER 'taxi_store.h5'
```

**NOTE** that if you run clustering twice (step 4) with the same day and order values, the existing clusters will be appended. This means you will have the clusters created in the original run plus the clusters in the second run. To ensure that this is not the case, the clusters must first be deleted. To delete all clusters you can you the `--delete` command, to delete the clusters produced on a certain run you can use `--delete_run`.
