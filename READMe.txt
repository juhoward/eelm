This directory includes python code and saved features to serve as a demonstration of the 
capabilities of the eeml_clustering method described in the Clustering Presentation.ppt. 

The required packages are included in the eeml_clusting.yaml file. You 
can install these into a conda environment with the following command:
```conda create --file eeml_clustering.yaml```

Two pickled data structures, `feature_set.pkl` and `UMAP_embedding.pkl` are given to save 
computational time. To produce similar results to those that are given, you can provide a path to 
the FLIR_ADAS thermal 8 bit training set to create features and a new UMAP embedding. To do so, use 
the following code 
```python eeml_style_identification.py --data_dir /path/to/data --load_features False --load_umap False```
