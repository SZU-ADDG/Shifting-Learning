# Shifting Learning: A Biologically Plausible Learning Paradigm for Spiking Neural Networks

This is the official code repository for the paper: Shifting Learning: A Biologically Plausible Learning Paradigm for Spiking Neural Networks

In our paper, we introduce:

- **A brain-inspired spiking learning system:** In this system, the information is transmitted in the form of spikes, and synaptic connection strengths are treated as realistic, timing-dependent variables called shifting timings.

<p align="center">
<img src="images/overall.png" width="75%" height="75%" />
</p>

- **Shifting learning algorithm:** The mechanism is based on three factors: the local affection from the STDP mechanism, the feedback signal transmitted through the feedback connection, and the global reward-punishment signal produced by the release of neuromodulators. It leverages a combination of biological signals to achieve performance comparable to same-scale spiking neural networks trained with gradient methods.

<p align="center">
<img src="images/plasticity.png" width="85%" height="85%" />
</p>

## Software & Hardware Requirements

The implementation is based on Pytorch and the required environmental dependencies are relatively simple. Version control tool is not necessary.

All the code is easy to run directly on CPU and common Nvidia GPUs. The experiments on simple datasets can be done on CPU in proper time but the ones on more complex data is recommended to use GPU.

## Details

Detailed descriptions of each .py file about what it is and how it can do is provided as follows:

**1. Dataset.py:** Preprocess the datasets used to train and test our model, including Iris, Spike, MNIST, and Fashion-MNIST. The raw datasets are provided here; please refer to the datasets folder. The Spike dataset, which includes the files "gamma_5-inhomogbgnoise-trials-250-syn-500-feat-9-train" and "gamma_5-inhomogbgnoise-trials-250-syn-500-feat-9-validation", is used for validating the temporal credit assignment (TCA) problem.

**2. SL_Layer.py & SL_MultiInputLy.py:** Our model, spiking neural network (SNN) equipped with shifting learning. There are two variants: (1) SL_Layer.py for input neurons receive a single spike from each synapse in a trial, and (2) SL_MultiInputLy.py for input neurons can receive multiple spikes from each synapse used in the TCA task.

**3. SL_xor.py:** Our model used to solve the XOR problem. It can be run once directly or executed multiple times to report the accuracy. The number of repetitions can be set manually.

**4. SL_small.py:** Our model used to solve the Iris classification task. It is designed as a platform that can be easily generalized to handle other small-scale benchmarks.

**5. SL_mnist.py:** Our model applied to the MNIST and Fashion-MNIST datasets, and it can be accelerated using GPUs.





