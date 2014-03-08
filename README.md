
Example of using nupic to recognize digits.

There is a set of labeled training data and a set of labeled test data.  You can also provide your own image and see if it recognizes it.

It uses the spatial pooler directly.

Here is the algorithm:

Training phase:

- For each image in the labeled training data 

- Present to spatial pooler

- Save the activecolumns returned by the spatial pooler in results dictionary, keyed on the image label (eg, "1")

Testing phase:

- For each image in the labeled training data 

- Present to spatial pooler and get activecolumns

- Look in results dictionary created during training phase for exact match of this activecolumns result.  print out recognized image label or error

