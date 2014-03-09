
Use the NuPIC Spatial Pooler to recognize digits.

The spatial pooler is trained on digits like:

![](https://github.com/tleyden/nupic-digitrecognizer/blob/master/data/training/9.png?raw=true)

and then tested with noisy version of the digits:

![](https://github.com/tleyden/nupic-digitrecognizer/blob/master/data/testing/9.png?raw=true)

# Training phase

- For each image in the labeled training data 

  - Present to spatial pooler

  - Save the activecolumns SDR returned by the spatial pooler in results dictionary, keyed on the image name (eg, "1.png")

# Testing phase

- For each image in the labeled testing data 

  - Present to spatial pooler and get activecolumns SDR

  - Get the saved activecolumns SDR for this image label (eg, "1.png") from the results dictionary from training step

  - Make sure that the SDR returned from the test image is an *exact match* with the training SDR

# Pre-requisites

* Install NuPIC
* Install Pillow

# Running

```
$ python digitrecognizer.py
```

It will throw an exception if any of the digits are not recognized in either testing set.

# Future extensions

* Implement spatial invariance (eg, if these digit inputs are shifted a few pixels then it won't match)

# Reference

* [Question regarding spatial poolers in context of digit recognition](http://lists.numenta.org/pipermail/nupic_lists.numenta.org/2014-March/003121.html)
* [Training on Handwritten Digit Dataset using CLA](http://lists.numenta.org/pipermail/nupic_lists.numenta.org/2013-July/000538.html)