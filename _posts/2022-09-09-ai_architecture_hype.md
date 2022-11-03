---
title: "Practically Beyond ‘Novel’ Methods"
date: 2022-09-09 00:00:00
layout: post
excerpt: Oloren AI using Oloren ChemEngine (aka Oloren Chem Engine, OCE) demonstrates that models constructed simply with OCE outperform graph neural networks (GNNs) including GEM, D-MPNN, AttentiveFP, GROVER, PretrainGNN, etc. on toxicity tasks (Tox21) part of the MoleculeNet suite of benchmarks. Molecular property predictors powered by ensembles of traditional chemoinformatics methods and modern neural networks are extremely powerful.
toc: false
author: David Huang
featured_image: https://images.unsplash.com/photo-1508188317434-1fd219bb636f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2293&q=80
---

We love graph neural networks and deep learning—that’s why we gave a [talk](https://oloren.ai/blog/acs_fall2022.html) recently on supervised contrastive learning on super-large PubChem-BioAssay dataset. But, when it comes to implementing these ‘novel’ methods into practice things get complicated. There’s a plethora of papers that sound like “AI Deep Learning Transformer Graph Neural Network with State-of-the-Art Quantum Geometric Representations Leveraging Big Federated Data and Provably Powerful Meta-Learning” or something like that, and I don’t even think I got all the buzzwords.

It’s a confusing landscape of optimizing for new methods (often with their own confusing repositories and documentation) and sticking to the tried-and-true. methods It often feels lose-lose. If you implement the new methods, it takes forever and ever and often at the end of it the new methods don’t even work. If you stick with the tried-and-true methods, the question “what if I implemented the new methods?” looms large.

That’s the bad news. The good news is that the **practical** application of AI should be and is a lot simpler than that. Let’s compare some new and old methods.

We read and really liked this paper on **“Geometry-enhanced molecular representation learning for property prediction” (Fang et al.)**, but they compared only against other graph neural networks and not “baseline” models like Random Forests and MLPs on chemical descriptors and fingerprints) and ensembled variants thereof. We found that those baselines performed BETTER than the proposed graph neural networks on Tox21 measured by AUC.

A random forest trained on the RDKit chemical descriptors provided by Descriptastorus (model1) had an AUC **0.780** vs the best graph neural network with AUC **0.781**. An ensembled model of gradient-boosted random forests (Descriptastorus’s rdkit2dnormalized and morgan3counts, and Mol2Vec) (model2) having an AUC of **0.789** performed better than all provided graph neural networks.


```python
model1 = oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators = 1000)
model2 = oce.BaseBoosting([oce.RandomForestModel(oce.DescriptastorusDescriptor("rdkit2dnormalized"), n_estimators = 1000),
                            oce.RandomForestModel(oce.DescriptastorusDescriptor("morgan3counts"), n_estimators = 1000),
                            oce.RandomForestModel(oce.Mol2Vec(), n_estimators = 1000)])
```


We haven’t done a full study on all the tasks presented and almost certainly there are many situations where GNNs are better (we use GNNs on plenty of situation-dependent occasions). We just wanted to punch-back at the overwhelming buzzword inflation that happens everywhere.

**Bottom Line: The newest shiniest thing isn’t always better. Be skeptical and ask for baselines. Practically, simple is often better.**

Lots of disclaimers and qualifications.

First, we found the geometry-enhance ML to be immensely helpful in both ideas (translating geometry to graph embeddings) and in that they systematically tried lots of graph neural networks. However, they used their own split which means that there is no way of comparing directly to the original publication of the other graph neural networks; furthermore, it was a bit of a hassle to completely recreate their split, which we verified by checking the class ratios—there was a ton of hidden complexity there.

Second, we find internally that we can ‘beat’ this ensembled random forests baseline by combining even more diverse models—including graph neural networks.

Third, we have an interest in the framework we did our benchmarking on as we provide it open-sourced at [https://github.com/Oloren-AI/olorenchemengine](https://github.com/Oloren-AI/olorenchemengine), and we also provide commercial licenses, extensions, and applications thereof.

Lastly, this isn’t to say we don’t like new AI approaches—we gave a whole presentation at ACS this Fall 2022 on supervised contrastive learning—just that we value practicality and baselines.

Final notes. This result shouldn’t be surprising: in the original D-MPNN paper they found very little difference between GNNs and RFs/MLPs on Tox21.

<script src="https://gist.github.com/davidzqhuang/1f47ab501f97dea5e6a9a365e2c256e9.js"></script>