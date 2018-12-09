

# Visualization - Imagenet VGG19 Forward Propagation of Layers

<br />

### Project Description

I used Python Tensorflow and Indesign to create a visualization of the recognizing processus of model "Imagenet-VGG19".

Thanks to the youtube video by "互联网开发教程", I learned an idea to analyse this model,

and then I visualized the Forward Propagation of all 36 layers of this model.

!: Be careful that these features I visualized are just the first one of all features, they are not the ones with the best "RELU" result!

<br />

### Library dependence

```python

    import os
    
    import numpy as np
    
    import tensorflow as tf
    
    import matplotlib.pyplot as plt
    
    import scipy.misc
    
    import scipy.io
 
```

<br />

### Deployment steps

 * You can download the .zip and the model form web, personnally I use Spyder. Sometimes you may change your working directory:

```python
    # Enter directory to change dir
    
    os.chdir('.') # Enter your directory
    
    #os.chdir('D:/Github/Practicing/visua_vgg19layers') #This one is my dir
```

 * .py will import test photo of cats automatically and export figures in the "output" folder


 * If you have Adobe Indesign, you can use my .indd to make the typesetting and designing :-)
 
<br />

### Version

2018/10/15

<br />

### Reference

< Model VGG19 >

http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat

< Model Analysing >

https://www.youtube.com/watch?v=QyWGHOqZH4k&index=15&list=PL8LR_PrSuIRhpEYA3sJ-J5hYGYUSwZwdS 
By 互联网开发教程

