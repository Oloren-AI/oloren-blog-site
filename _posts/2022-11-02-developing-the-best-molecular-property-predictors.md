---
title: Developing the ‘best’ molecular property predictors
date: 2022-11-02 00:00:00 Z
layout: post
excerpt: We are releasing a new site and platform for benchmarking molecular property
  predictors across many open benchmarks.
toc: false
author: David Huang
---

### Benchmarking efforts and site

We recently released a set of benchmarks, [topping the leaderboards](https://github.com/Oloren-AI/OCE-TDC) on Therapeutic Data Commons, and are releasing a website as well: benchmarks.chemengine.org. On that site, we will benchmark models created by Oloren ChemEngine on relevant datasets. We are still in the process of migrating past experiments.

We are releasing all this—out into the open—because we believe that in Oloren ChemEngine (OCE, [https://github.com/Oloren-AI/olorenchemengine](https://github.com/Oloren-AI/olorenchemengine "https://github.com/Oloren-AI/olorenchemengine")) we have the best system for molecular property prediction and we aren’t afraid to back up that claim. We also believe that publishing our benchmark results out there is a great way to push ourselves to improve our models.

### Reproducibility of benchmarks

Before I dive into a longer story on our modeling strategy, I want to emphasize one super cool feature about this benchmarking site: with OCE installed, you can copy-paste the “Model Parameters” JSON dictionary for any model you see on the site, and 100% accurately reproduce that model locally on your own machine. No fussing about, no complex reimplementation and integrations: it’s a game-changer for reproducibility.

We began our modeling journey with an ambitious project: we’d just seen the publication of an image recognition paper on “Supervised Contrastive Learning”, a way to extract representation from labeled databases, and we wanted to apply it to PubChem BioAssay. So we did, spending months on the process, scaling up infrastructure to handle massive quantities of data to train a new learned molecular representation we call [OlorenVec](https://oloren.ai/blog/acs_fall2022.html).

This worked great! But, when it came to integrating OlorenVec into a project with other models, it quickly became clear how fragmented the AI molecular property prediction ecosystem was. After a month or so of stringing together scripts and Jupyter notebooks, we realized we needed a solution. So, we set out to build what we now call Oloren ChemEngine.

### Oloren ChemEngine design

We had a few key design requirements for this library: (1) we could easily save, load, parameterize and recreate models, and (2) we could easily integrate different modeling solutions together, whether that be ensembling models together, or recombining different molecular representations and learners. And, we built this. We learned later that many pharma companies had similar-ish systems in-house.

After the framework was created, we shamelessly went hunting to find the best algorithms for property prediction in the literature, and just kept adding those algorithms into OCE! We like to say that we know our models are the best because if they aren’t, we just copy the ones which are better. Of course, we add model ensembling on top and a few other in-house algorithms to catapult our models into being the best.

### Questions for the Future

We’ve had the opportunity to drill deep into the capabilities of OCE in specific projects, but we haven’t really had the chance to explore the general space of prediction problems. Some specific questions we are interested in include the following.

Are there specific sets of hyperparameters that work universally well?

What is the best way to ensemble models?

What are the best molecular representations?

Are specific types of models better suited for specific types of datasets?

How much room is left for algorithmic improvement and how it is just due to “scientific uncertainty”?

### Using [benchmarks.chemengine.org](http://benchmarks.chemengine.org/)

There are, of course, a whole other set of questions related to practicality and adoption, including research and development into interpretability and uncertainty quantification (a topic for another day). The goal of benchmarks.chemengine.org though is to function both like a recipe book for molecular property prediction and like a boxing ring.

It’s like a recipe book in that we can copy-paste the best models from the site to use in our own modeling projects. It’s a way to be confident that the models you’re using are the best that can be made. It’s like a boxing ring in that we can directly compare two models directly and reproducibly and find out which is better and over what circumstances.