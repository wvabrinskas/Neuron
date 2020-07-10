# Neuron

## Introduction
Neuron is a swift package I developed to help learn how to make neural networks. It is far from perfect and I am still learning. There is A LOT to learn here and I've just scratched the surface. As of right now this package provides a way to get started in machine learning. It allows for multiple input and outputs as well as customizing the number of nodes in the hidden layer. As of right now you can create ONE hidden layer, this will change in the future. 

#### Disclaimer
This is very much a `BETA` project and to only be used as a learning tool as of now.

#### Support 
- [Twitter](https://twitter.com/wvabrinskas)

Feel free to send me suggestions on how to improve this. I would be delighted to learn more!! You can also feel free to assign issues here as well. 

#### Resources 

- https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f?gi=5b433900266a
- https://towardsdatascience.com/how-does-back-propagation-in-artificial-neural-networks-work-c7cad873ea74
- https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
- https://github.com/nature-of-code/noc-examples-processing/blob/master/chp10_nn/NOC_10_01_SimplePerceptron
- [Make Your Own Neural Network - Tariq Rashid](https://www.amazon.com/Make-Your-Own-Neural-Network-ebook/dp/B01EER4Z4G)


## Implementation

- Sample Project - https://github.com/wvabrinskas/Swift-Neural-Network

### The Brain
It is fairly simple to setup the neural network `Brain`. This will be the only object you interface with. 

#### Initialization
```  
  private lazy var brain: Brain = {
    let nucleus = Nucleus(learningRate: 0.05, bias: 0.01, activationType: .sigmoid)
    return Brain(inputs: 4, outputs: 4, hidden: 4, nucleus: nucleus)
  }()
```
- The `Brain` object takes in 4 properties, `inputs`, `outputs`, `hidden`, and `nucleus`
    - `inputs` - the number of input nodes
    - `outputs` - the number of output nodes
    - `hidden` - the number of hidden nodes in the single hidden layer
    - `nucleus` - a `Nucleus` object defining the learning properties of each node

- The `Nucleus` object takes in 3 optional properties `learningRate`, `bias`, and `activationType`
    - `learningRate` - how quickly the node will adjust the weights of its inputs to fit the training model 
        - Usually between `0` and `1`. 
        - Default `0.1`
    - `bias` - the offset of adjustment to the weight adjustment calculation. 
        - Usually between `0` and `1`. 
        - Default `0.1`
    - `activationType` - an enum defining the activation equation to use for the system. Options are: 
        - `.reLu`
        - `.leakyRelu`
        - `.sigmoid`
