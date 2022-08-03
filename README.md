<p align="center">
<img width="500" src="images/neuron.png"> 
</p>

![](https://img.shields.io/github/v/tag/wvabrinskas/Neuron?style=flat-square)
![](https://img.shields.io/github/license/wvabrinskas/Neuron?style=flat-square)
![](https://img.shields.io/badge/swift-5.6-orange?style=flat-square)
![](https://img.shields.io/badge/iOS-13+-darkcyan?style=flat-square)
![](https://img.shields.io/badge/macOS-11+-darkcyan?style=flat-square)
![](https://img.shields.io/badge/watchOS-6+-darkcyan?style=flat-square)
![](https://img.shields.io/badge/tvOS-13+-darkcyan?style=flat-square)

[![Tests](https://github.com/wvabrinskas/Neuron/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/wvabrinskas/Neuron/actions/workflows/tests.yml)

# Support 
[<img width="75" src="images/twitter.png"> ](https://twitter.com/wvabrinskas)
[<img width="75" src="images/discord.png">](https://discord.gg/p84DYph4PW)
[<img width="75" src="images/www.png">](https://williamvabrinskas.com)

Feel free to send me suggestions on how to improve this. I would be delighted to learn more!! You can also feel free to assign issues here. Run the unit tests as well to learn how the project works!

[Full Documentation](https://williamvabrinskas.com/Neuron/documentation/neuron/)

# Grand Re-Opening! 

Version 2.0 of Neuron is here! This new version of Neuron is a complete rewrite from the ground up of the architecture. It is much more streamlined, with faster execution. Its usage also aligns more with commonly used ML frameworks like [Keras](https://keras.io/) and [PyTorch](https://pytorch.org/). 

# Background

Neuron has been a pet project of mine for years now. I set off to learn the basics of ML and I figured the best way to learn it was to implement it myself. I decided on Swift because it was the language I knew the most and I knew it would be challenging to optimize for ML as it has a lot of overhead. What you're seeing here in this repository is an accumulation of my work over the past 2 years or so. It is my baby. I decided to make this open source as I wanted to share what I've learned with the ML and Swift community. I wanted to give users of this framework the opportunity to learn and implement ML in their projects or apps. Have fun! 

There is still a lot missing in this framework but with this rewrite I brought a lot more flexibity to the framework to allow for playing around with different architectures and models. There are some example models provided with the framework, like Classifier, GAN, WGAN, and WGANGP. I am always working on this project and will continue to provide updates. 

# Contribution Policies
## Filing Issues
Feel free to file issues about the framework here or contact me through the [Discord](https://discord.gg/p84DYph4PW). I am open to all suggestions on how to improve the framework. 

## Pull Requests
There are automated tests that run when a PR is created to the `develop` or `master` branches. These tests must pass before a PR can be merged. All PRs must merge into the `develop` branch. 
## Branching
All features must be branched off the `develop` branch. 

# Usage
## Tensor
The main backbone of Neuron is the `Tensor` object. This object is basically a glorified 3D array of numbers. All `Tensor` objects are 3D arrays however they can contain any type of array in-between.