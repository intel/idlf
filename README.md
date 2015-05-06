The Intel® Deep Learning Framework
==================================


The Intel® Deep Learning Framework (**IDLF**) is a SDK library for Deep Neural Networks  training and execution.

It includes the API that enables building neural network topology as a compute workflow, functions to optimize the graph and execute it on hardware. Our initial focus is neural network driven object classification (ImageNet topology) implemented on CPU (Xeon) and GPU (Gen). 

The API is designed in the way allowing us to easily support more devices in the future. Our key principle is achieving maximum performance on each supported Intel platform.


----------

#### This Project Provides: 

The Intel® Deep Learning Framework provides a unified framework for Intel® platforms accelerating Deep Convolutional Neural Networks. 

----------

#### Key reasons to use IDLF: ####

- **Heterogeneous**: Supports a wide array of Intel® accelerators from CPUs to GPUs and more
- **Performance**: We’ve optimized the underlying code so you don’t have to
- **Scalability**: Plans for scaling workloads across multiple nodes perfect for cloud deployments
- **Configurability**: There isn’t just one type of Neural Network.  You may support them all using a single software platform
- **Concurrent Classification & Learning**: Support for incremental learning at runtime, which allows for more flexible and general networks

----------

#### The Value of the IDLF Project ####

IDLF is an open source project providing a layered, heterogeneous framework for accelerated classification and training on Intel® hardware targeted at cloud environments. 
IDLF provides support for APIs spanning any level of abstraction:

- Network Primitives (convolution, ReLU, etc.)
- Device level (CPU, GPU, etc.)
- Node level (Single Server)
- Cluster level (Multiple Servers)

----------

#### Who’s It For ####

The IDLF project can be used by application developers, cloud service providers, academic researchers, or anyone seeking the best performance on the full spectrum of Intel® platforms for Deep Learning. 
Support for additional hardware platforms can be added over time in a completely API-agnostic manner allowing for maximum code reuse. 

----------

#### Project Specifics ####

This is an active open source project distributed under the BSD 3-clause open source license.

- IDLF is developed in C++ with C APIs
- Supported OSes: Linux*, Microsoft* Windows*
- Supported C++ compilers: Visual Studio* 2013, GCC and ICC 15
- CPU implementation depends on AVX2 instruction set (Haswell and newer CPU architectures)
- GPU implementation depends on OpenCL driver
The current release should be treated as an object classification performance and efficiency demo.
Implementation of the CaffeNet topology from Caffe Model Zoo ([http://caffe.berkeleyvision.org/model_zoo.html](http://caffe.berkeleyvision.org/model_zoo.html)) is provided as the demo application.

----------

#### About Intel® Involvement ####

Intel® is the leading contributor to IDLF, enabling application and system developers to make the most of Intel® Xeon™ and Xeon Phi™ processors as well as Intel® Iris™ Pro graphics.