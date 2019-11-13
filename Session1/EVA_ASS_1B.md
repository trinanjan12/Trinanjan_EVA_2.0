## Assignment 1B

### 1. Channels & kernels:

#### Channels: 

In case of image channel means different components of a single image. Channel refers to the component of image that stores similar information. Digital images have 3 dimensions height, weight & depth. The depth refers to the channels of that image.

A grey scale image has only one colour. Hence it has only one channel.

RGB has three colours red, green & blue.

RBGD has 4 channels(dept with rgb)

#### Kernels: 

Kernels or filters are nothing but matrices which are used in image processing & convolution neural networks for extracting features from images.Therefore, they are also called feature extractor. Here feature of an image implies edges, curves, textures, patterns, parts of object etc. There are various filters used in image processing & each of them serves various purpose. For example edge detector, blur kernel, sharpen kernel, identity kernel etc. In digital image processing these kernels or matrices can be hand coded. But in case of convolution neural network these kernels are optimised by the network algorithm called back propagation & gradient descent without any manual intervention.  
​               

In case of convolution neural network the initial layers uses filters/kernels which can detect edges, textures & patterns. Gradually these filters gets complex when goes deeper into the network & eventually the kernels can detect objects.

### 2.  3 * 3 kernels are widely used: 

1. NVIDIA GPUs are mainly optimised for 3*3 kernels. Hence, using 3 * 3 kernel in convolution neural net helps the model to train faster thereby reduces computation time.

2. The other & important reason for using 3 * 3 kernels are reduced number of parameters/weights than other kernels(5 * 5 or 7 * 7). In case of 3 * 3 kernel 9 multiplication happens for each convolution step.

The reason is for 2 * 2 kernel there is no middle point. But for 3 * 3 we have centre point for which we can imagine a symmetric line.


### 3. How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199  

We have an input image of size 199x199 & we are performing 3x3 convolution to reach output size 1x1.

If we apply 3x3 convolution on an image the output dimensions are reduced by [199-(199-3+1)]=2. After 1st convolution the size will be 197x197 & then 195x195, 193x193, 191x191 and so on till 1x1.

Let’s assume the no of convolution required is n. Then,

199 = 1 + (n-1) x 2 => n =100. This means 100 convolution layers are required to reach 1x1 output.

Now, let’s validate the above calculation.

##### 199x199  |  197x197  |  195x195  |  193x193  |  191x191  |  189x189  |  187x187  |  185x185  | 183x183 |

##### 181x181  |  179x179  |  177x177  |  175x175  |  173x173  |  171x171  |  169x169  |  167x167  |  165x165 |

##### 163x163  |  161x161  |  159x159  |  157x157  |  155x155  |   153x153  |  151x151  |  149x149  |  147x147 | 

##### 145x145  |  143x143  |  141x141  |  139x139  | 137x137  |  135x135  |  133x133  |  131x131  |  129x129 |

##### 127x127  |  125x125  |  123x123  | 121x121  |  119x119  |  117x117  |  115x115  |  113x113  |  111x111  |

##### 109x109  |  107x107  |  105x105  |  103x103  |  101x101  |  99x99  |  97x97  |  95x95  |   93x93  | 

##### 91x91  | 89x89  |  87x87  |  85x85  |  83x83  |  81x81  |  79x79  |  77x77  |  75x75  |  73x73  |  71x71  | 

##### 69x69  |  67x67  |  65x65  |  63x63  |  61x61  |  59x59  |  57x57  |  55x55  |  53x53  |  51x51  |  49x49  |  

##### 47x47  |  45x45  |  43x43  |  41x41  |  39x39  |  37x37  |  35x35  |  33x33  |  31x31  |  29x29  |  27x27  | 

##### 25x25  |  23x23  |  21x21  | 19x19 | 17x17  |  15x15  |  13x13  |  11x11  |  9x9  |  7x7  |  5x5  |  3x3  | 1x1



It requires 100 convolutions to perform to get output of 1x1 from input image 199x199. 



 



