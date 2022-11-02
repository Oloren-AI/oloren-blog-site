---
title: Visualizing the chemical space
date: 2022-06-29 00:00:00 Z
layout: post
excerpt: After creating models for drug classification, we need a practical way of
  visualizing our chemical space and verifying our results. This guide will walk you
  through how to generate an interactive plotly graph of chemicals that renders 2D
  images of molecules on hover. A good library for this is molplotly, and the below
  tutorial teaches you how to write and customize your own code for this.
toc: false
author: Melissa Du
featured_image: https://oloren-blogcontent.s3.us-east-2.amazonaws.com/visualizing/fig4.png
---

We will cover two different types of chemical representations (Morgan fingerprint and RDKit 2D molecular descriptors), two different techniques for dimension reduction (PCA and t-SNE), generating a scatterplot in plotly, and creating an interactive Dash app which displays structures on hover. We'll be using the BACE dataset from MoleculeNet. All of the code is located in our “Putting Everything Together” section at the end, but we recommend reading through the sections to fully understand how everything works.

Let's get started! 🙂

# Chemical representations
### Morgan Fingerprints

RDKit provides a function for converting SMILES into Morgan fingerprints that requires the fingerprint radius and number of bits. It is common to use a radius of 2 or 3 and at least 1024 bits. Higher bit values will allow you to retain more information about the molecule.
In order to perform PCA or t-SNE, we convert the fingerprint into a 1d numpy array.

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np

def fp_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) # generate MF as bit vector
    fp = np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0') # convert bit vector to 1d numpy array
    return fp
```

### RDKit 2D Normalized Descriptor
[Descriptastorus](https://github.com/bp-kelley/descriptastorus) is a package for that allows us to efficiently generate a wide range of molecular descriptors from chemical SMILES. Luckily for us, they are also able to calculate and normalize the 2D molecular descriptors provided in RDKit.

The output of processing a SMILES on the RDKit2DNormalized generator is a 1d array.

```python
from rdkit import Chem, DataStruct
from descriptastorus.descriptors import rdNormalizedDescriptors

def rdnd_from_smiles(smiles):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    rdnd = generator.process(smiles)
    return rdnd
```

# Dimensionality Reduction
### Principal Component Analysis
The scikit-learn package provides a function to perform PCA. We just need to pass in our dataset as an array of the molecules’ chemical representations (chem_rep_list) and specify the number of components. To plot this on a graph, we are limited to 2 or 3 components.

```python
from rdkit import Chem, DataStructs
from sklearn.decomposition import PCA

def pca_df(chem_rep_list):
    pca = PCA(n_components=2)
    pca_arr = pca.fit_transform(chem_rep_list)
        return pd.DataFrame(pca_arr, columns=["Component 1", "Component 2"])
```

### t-Distributed Stochastic Neighborhood Embedding
Alternatively, we can use t-SNE to reduce the number of dimensions. However, this technique is more computationally heavy, so it’s recommended to use PCA first to reduce the data down to around 50 dimensions.

The t-SNE function has many [parameters](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). We recommend modifying them within a range as different values can give dramatically different results.

```python
from rdkit import Chem, DataStructs
from sklearn.manifold import TSNE

def tsne_df(chem_rep_list):
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(chem_rep_list)
    pca_50_variance = pca_50.explained_variance_ratio_
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca_result_50)mol = Chem.MolFromSmiles(smiles)
    return pd.DataFrame(tsne_results, columns=["TSNE-1", "TSNE-2"])
```

### Graphing with Plotly
We can use any of the chemical representations above with any of the dimension reduction techniques. Let’s combine some of the code we’ve written to create a function that allows us to generate a plotly graph.

```python
from rdkit import Chem, DataStructs
from sklearn.manifold import TSNE

def graph_chemical_space(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    color_col: str = None,
    id_col: str = None,
    chem_rep: str = "rdkit",
    dim_reduction: str = "TSNE",
    graph_title: str = None
):
    """
    Written by Melissa Du, Oloren AI
    df : pandas.DataFrame object
        a pandas dataframe that contains the data plotted in fig.
    smiles_col : str, optional
        name of the column in df containing the smiles plotted in fig (default 'SMILES').
        If provided as a list, will add a slider to choose which column is used for rendering the structures.
    color_col : str, optional
        name of the column in df that will specify the point colors plotted in fig (default None)
    id_col : str, optional
        name of the column in df that will specify id of the points plotted in fig (default None)
    chem_rep : "rdkit" | "mf" | "oc" , optional
        name of the desired chemical representation
    dim_reduction : "PCA" | "TSNE" , optional
        name of desired dimension reduction technique
    graph_title : str, optional
        title of graph
    """

    assert chem_rep in ['rdkit', 'mf', 'oc'], "Please enter a valid chemical representation"
    assert dim_reduction in ["PCA", "TSNE"], "Please enter a valid dimension reduction technique"
    assert smiles_col in df, "The specified smiles_col is not in the dataframe"

    funcs = {
        "rdkit" : rdnd_from_smiles,
        "mf" : fp_from_smiles,
        "TSNE": tsne_df,
        "PCA": pca_df
    }

    mol_list = []
    # convert smiles to desired chemical representation
    for smiles in df[smiles_col]:
        mol_list.append(funcs[chem_rep](smiles))
    # apply desired dimension reduction technique
    rd_df = funcs[dim_reduction](mol_list)
    # populate dataframe with color and id columns
    rd_df = rd_df.join(df[smiles_col])
    if id_col is not None:
        rd_df = rd_df.join(df[id_col])
    if color_col is not None:
        rd_df = rd_df.join(df[color_col])

    fig = px.scatter(rd_df, x='Component 1', y='Component 2', color=color_col, title=graph_title)
    return fig
```