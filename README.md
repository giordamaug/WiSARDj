# WiSARDj.jl

This package is a Julia implementation of WiSARD 
weightless neural network model as a classification method.

# Table of Contents

- [WiSARDj.jl](#wisardjjl)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [API](#api)
  - [MLJ support](#mlj-support)
  - [ScikitLearnBase support](#scikitlearnbase-support)
- [Example](#example)
- [History](#history)
- [See also](#see-also)

# Installation

Please note that the package is still under development and it actually is not registered in the Julia official package archive. 
Then in order to install the package you need to run the command:

To add the package from the Julia PL:
```julia
using Pkg
Pkg.add("https://github.com/giordamaug/WiSARDj")
```


These ccommands will install all `WiSARDj` required dependencies:

- `Distributed`
- `MLJBase`
- `ProgressMeter`
- `Random`
- `ScikitLearnBase`

## API

WiSARDj provides interface, i.e. `fit! ` and `provide` methods, according to the MLJ API.

## MLJ support

The MLJ interface of WiSARDj can be imported and used with commands

```julia
julia> using .WiSARDj.MLJInterface: WiSARDClassifier
julia> using MLJBase
julia> model = WiSARDClassifier(...)
julia> MLJ.fit!(model, X, y)
...
```

## ScikitLearnBase support

The ScikitLearnBase interface of WiSARDj can be imported and used with commands:

```julia
julia> using .WiSARDj.SciLearnInterface: WiSARDClassifier
julia> using ScikitLearnBase.fit!
julia> model = WiSARDClassifier(...)
julia> ScikitLearnBase.fit!(model, X, y)
...
```
# Example

We provide scripts for testing both interfaces (`MLJ` and `ScikitLearnBase`) in the `test`folder.
The scripts are classification examples of a train-test split of the `iris` dataset which is include in the `datasets` dir.

To run the tests you need these additional packages installed on your syste.
- `CSV` for dataset reading;
- `DataFrames` for dataset reading
- `MLJ` for data partition into train-test split

The scripts can be run by the shell commands:

```shell
> julia testScikitLearn.jl
> julia testMLJ.jl
```
or from the julia REPL:

```julia
julia> include("test/testScikitLearn.jl")
julia> include("test/testMLJ.jl")
```

In the `notebooks`folder there are  two jupyter notebooks reproducing the execution
of the two test above mentioned scripts for the MLJ and ScikitLearn APIs.

# History

The WiSARD weightless neural network model, also known as 
RAMnets, is the basis of the first patented commercial product of a nartificial neural network machine (Wilkie, Stonham and Aleksander Recognition Device).

WiSARD was originally conceived as a pattern recognition device mainly focusing on image processing domain.
With ad hoc data transformation, WiSARD can also be used successfully as multiclass classifier in machine learning domain.

The WiSARD is a RAM-based neuron network working as an <i>n</i>-tuple classifier.
A WiSARD is formed by as many discriminators as the number of classes it has to discriminate between. 
Each discriminator consists of a set of <i>N</i> RAMs that, during the training phase, l
earn the occurrences of <i>n</i>-tuples extracted from the input binary vector (the <i>retina</i>).

In the WiSARD model, <i>n</i>-tuples selected from the input binary vector are regarded as the “features” of the input pattern to be recognised. It has been demonstrated in literature [14] that the randomness of feature extraction makes WiSARD more sensitive to detect global features than an ordered map which makes a single layer system sensitive to detect local features.

More information and details about the WiSARD neural network model can be found in Aleksander and Morton's book [Introduction to neural computing](https://books.google.co.uk/books/about/An_introduction_to_neural_computing.html?id=H4dQAAAAMAAJ&redir_esc=y&hl=it).

The WiSARD4WEKA package implements a multi-class classification method based on the WiSARD weightless neural model
for the Weka machine learning toolkit. A data-preprocessing filter allows to exploit WiSARD neural model 
training/classification capabilities on multi-attribute numeric data making WiSARD overcome the restriction to
binary pattern recognition.

For more information on the WiSARD classifier implemented in the WiSARD4WEKA package, see:

> Massimo De Gregorio and Maurizio Giordano (2018). 
> <i>An experimental evaluation of weightless neural networks for 
> multi-class classification</i>.
> Journal of Applied Soft Computing. Vol.72. pp. 338-354<br>

# See also

- M. Morciniec and R. Rohw, [The n-tuple Classier: Too Good to Ignore](http://www.haralick.org/ML/NCRG_95_013.pdf), *1995*

- I. Aleksander, M. D. Gregorio, F. França, P. Lima, H. Morton, [A brief introduction to Weightless Neural Systems](https://www.semanticscholar.org/paper/A-brief-introduction-to-Weightless-Neural-Systems-Aleksander-Gregorio/25a367c108745dbc3c3729e683b645d09c6dd23b), published din *The European Symposium on Artificial Neural Networks 2009* .



