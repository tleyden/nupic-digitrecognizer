
from nupic.research.spatial_pooler import SpatialPooler as SP
from PIL import Image
import numpy as np
import os
import sys

class DigitRecognizer(object):
      
    def __init__(self, trainingDataDir, testingDataDir):
        self.trainingDataDir = trainingDataDir
        self.testingDataDir = testingDataDir
        self.inputShape = (28, 28)
        self.columnDimensions = (56, 56)
        self.columnNumber = np.array(self.columnDimensions).prod()
        self.inputSize = np.array(self.inputShape).prod()
        self.spatialPooler = self._initSpatialPooler()

    def _initSpatialPooler(self):

        '''
      potentialRadius = 10000, # Ensures 100% potential pool
      potentialPct = 1, # Neurons can connect to 100% of input
      globalInhibition = True,
      numActiveColumnsPerInhArea = 1, # Only one feature active at a time
      # All input activity can contribute to feature output
      stimulusThreshold = 0,
      synPermInactiveDec = 0.01,
      synPermActiveInc = 0.1,
      synPermConnected = 0.1, # Connected threshold
      maxBoost = 3,
      seed = 1956, # The seed that Grok uses
      spVerbosity = 1)

        '''

        print "Creating spatial pooler .. go check twitter"

        spatialPooler = SP(
            self.inputSize,   
            self.columnDimensions,
            potentialRadius = self.inputSize,
            numActiveColumnsPerInhArea = int(0.02*self.columnNumber),
            globalInhibition = True,
            synPermActiveInc = 0.01
        )

        # experimenting with different parameters
        spatialPoolerAlternative = SP(
            self.inputSize,   
            self.columnDimensions,
            # potentialRadius = self.inputSize,
            potentialRadius = 10000, # Ensures 100% potential pool
            potentialPct = 1, # Neurons can connect to 100% of input

            numActiveColumnsPerInhArea = int(0.02*self.columnNumber),
            globalInhibition = True,
            # All input activity can contribute to feature output
            stimulusThreshold = 0,
            synPermInactiveDec = 0.01,
            synPermActiveInc = 0.1,
            synPermConnected = 0.1, # Connected threshold
            maxBoost = 3
            # synPermActiveInc = 0.01
        )
        print "Spatial pooler is operational."
        return spatialPooler


    def run(self):
        '''
        Run the digit recognizer 
        '''
        trainingResults = self._train()
        print "Training results: %s" % trainingResults
        testingResults = self._test(trainingResults)
        print "Testing results: %s" % testingResults
    
    def _train(self, numIterations=80):

        trainingResults = {}
        for i in xrange(numIterations):
            self._trainIteration(trainingResults)
        return trainingResults

    def _trainIteration(self, trainingResults):

        # - For each image in the labeled training data 
        # - Present to spatial pooler
        # - Save the activecolumns returned by the spatial pooler in 
        #   results dictionary, keyed on the image label (eg, "1")
        for filename in os.listdir(self.trainingDataDir):
            if not filename.endswith("png"):
                continue

            filenameWithPath = os.path.join(self.trainingDataDir, filename)
            print filenameWithPath

            activeColumns = self._runSpatialPoolerOnFile(filenameWithPath)

            trainingResults[filename] = activeColumns

            print "done spatial pooler"


    def _runSpatialPoolerOnFile(self, filenameWithPath):
        image = Image.open(filenameWithPath)
        inputArray = self._convertToInputArray(image)
        activeColumns = np.zeros(self.columnNumber)
        
        print "Calling spatial pooler compute() with input: "
        self._prettyPrintInputArray(inputArray)
        
        self.spatialPooler.compute(inputArray, True, activeColumns)
        print "called compute(), activeColumns:" 
        print activeColumns.nonzero()
        return activeColumns


    def _test(self, trainingResults):
        
        testingResults = {}

        # - For each image in the labeled testing data 
        # - Present to spatial pooler
        # - Look in results dictionary created during training phase for exact match of this 
        #   activecolumns result.  print out recognized image label or error

        for filename in os.listdir(self.testingDataDir):
            if not filename.endswith("png"):
                continue

            filenameWithPath = os.path.join(self.testingDataDir, filename)
            print filenameWithPath

            activeColumns = self._runSpatialPoolerOnFile(filenameWithPath)

            for key, savedActiveColumns in trainingResults.items():
                if ((activeColumns == savedActiveColumns).all()):
                    testingResults[key] = True 
                else:
                    testingResults[key] = (activeColumns == savedActiveColumns)

        return testingResults


    def _prettyPrintInputArray(self, inputArray):
        reshaped = inputArray.reshape(28, 28)
        print reshaped

    def _convertToInputArray(self, image):

        image = image.convert('1')  # Convert to black and white

        inputArray = np.zeros(self.inputSize, np.int8)

        # is there a slicker way to get the image data into a numpy array?
        
        imageData = image.getdata()

        i = 0
        for pixel in imageData:
            if pixel == 0:
                inputArray[i] = 1
            i += 1

        return inputArray



if __name__ == "__main__":
    _trainingDataDir = os.path.join("data", "training")
    _testingDataDir = os.path.join("data", "testing")
    digitRecognizer = DigitRecognizer(_trainingDataDir, _testingDataDir)
    digitRecognizer.run()

