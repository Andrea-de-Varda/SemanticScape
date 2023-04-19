# SemanticScape
Code to construct SemanticScape vectors, i.e., distributional semantic vectors grounded in distance patterns between objects in natural images.

## Motivation
SemanticScape is a semantic model grounded in the visual relationships between objects in natural images. It captures the latent statistics of the spatial organization of objects in the visual environment, aiming to fill the gap between distributional semantics, where concepts are represented on the basis of text-based co-occurrence statistics, and convolutional neural networks, which represent concepts as averaged image-based sub-symbolic features. Its implementation is based on the calculation of the summed Euclidean distances between all object pairs in visual scenes, which are then abstracted by means of dimensionality reduction (Singular Value Decomposition). 

SemanticScape is effective in capturing human explicit intuitions on semantic and visual similarity, relatedness, analogical reasoning, and several semantic and visual implicit processing measurements.

## Model
Our model is based on an abstraction over the distributional statistics of the objects' locations in natural images. Our model is trained on [Visual Genome](https://link.springer.com/article/10.1007/s11263-016-0981-7), a dataset specifically designed to foster AI research in cognitive tasks.

We start by localizing the objects in the images as the centroid of the parallelogram of its bounding box. Starting from these coordinates, we calculate all the pairwise distances between the objects in the image (see the figure below). We then constructed a squared Euclidean distance matrix between all the object pairs coordinates. All the matrix entries corresponding to objects in the VG vocabulary that were not present in $v$ were set to 0. 

![alt text](https://github.com/Andrea-de-Varda/SemanticScape/blob/main/figures/img_with_net.png?raw=true)

<img src="https://github.com/Andrea-de-Varda/SemanticScape/blob/main/figures/img_with_net.png?raw=true" width="100" height="100">

