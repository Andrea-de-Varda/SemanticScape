# SemanticScape
Code to construct SemanticScape vectors, i.e., distributional semantic vectors grounded in distance patterns between objects in natural images.

## Motivation
SemanticScape is a semantic model grounded in the visual relationships between objects in natural images. It captures the latent statistics of the spatial organization of objects in the visual environment, aiming to fill the gap between distributional semantics, where concepts are represented on the basis of text-based co-occurrence statistics, and convolutional neural networks, which represent concepts as averaged image-based sub-symbolic features. Its implementation is based on the calculation of the summed Euclidean distances between all object pairs in visual scenes, which are then abstracted by means of dimensionality reduction (Singular Value Decomposition). 

SemanticScape is effective in capturing human explicit intuitions on semantic and visual similarity, relatedness, analogical reasoning, and several semantic and visual implicit processing measurements.

## Pre-trained models
In this repository, you can find the 2-, 3-, 5-, 10-, 15-, 20-, 25-, 30-, and 50-dimensional models saved as pickle files. In our analyses, the best-performing model was the 25-dimensional one. The models can be downloaded from the folder `trained-models`, and uploaded with few lines of code: 

```python
import pickle

with open("vizgen_rank_25D", 'rb') as handle:
    d_25d = pickle.load(handle)
```

You might want to use our model to calculate the SemanticScape-based similarity between two words or concepts. To do so, the metric we recommend is the cosine similarity:

```python
import numpy as np

def cosine(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

print(cosine(d_25d["dog"], d_25d["leash"]))
```

The pre-trained 25-dimensional model is also available in csv (compressed; `trained-models/SemanticScape_25D.zip`) and json format (`trained-models/d_25d.json`). This might be useful if you are not working in Python. 

If you are interested in the models of higher dimensionality (which, by the way, did not perform very well), please open an issue and I will share them with you. 

## Model

Locating objects           |  Calculating distances
:-------------------------:|:-------------------------:
![](https://github.com/Andrea-de-Varda/SemanticScape/blob/main/figures/img_with_labels.png?raw=true)  |  ![](https://github.com/Andrea-de-Varda/SemanticScape/blob/main/figures/img_with_net.png?raw=true)


Our model is based on an abstraction over the distributional statistics of the objects' locations in natural images. Our model is trained on [Visual Genome](https://link.springer.com/article/10.1007/s11263-016-0981-7), a dataset specifically designed to foster AI research in cognitive tasks.

We start by localizing the objects in the images as the centroid of the parallelogram of their bounding box. Starting from these coordinates, we calculate all the pairwise distances between the objects in the image (see the figure above). We thus construct a squared Euclidean distance matrix between all the object pairs coordinates. We convert distances to ranks, and decompose the matrix through Singular Value Decomposition. 

### Validation
We validated our model against several cognitive benchmarks. We list here the sources for those who wish to replicate our results:

- :eyes: Visual benchmarks:
  - ViSpa concept representations ([data](https://sites.google.com/site/fritzgntr/software-resources/vispa))
  - ViSpa test data ([paper](https://www.researchgate.net/publication/355415433_ViSpa_Vision_Spaces_A_computer-vision-based_representation_system_for_individual_images_and_concept_prototypes_with_large-scale_evaluation)|[data](https://osf.io/qvw9c)).
- :speech_balloon: Semantic benchmarks:
  - word2vec representations ([paper](https://arxiv.org/abs/1301.3781)|[data](https://github.com/Unipisa/DSMs-evaluation))
  - SimLex-999 taxonomic similarity ([paper](https://arxiv.org/abs/1408.3456v1)|[data](https://fh295.github.io/simlex.html)).
  - Evocation judgements ([paper](https://link.springer.com/chapter/10.1007/978-3-642-22613-7_5)|[data](http://wordnet.cs.princeton.edu/downloads.html))
  - Semantic Priming Project ([paper](https://link.springer.com/article/10.3758/s13428-012-0304-z)|[data](https://osf.io/n7gqa/))
- :twisted_rightwards_arrows: Analogy
  - Bigger Analogy Test Set ([paper](https://aclanthology.org/N16-2002.pdf)|[data](https://u.pcloud.link/publink/show?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV)) 

