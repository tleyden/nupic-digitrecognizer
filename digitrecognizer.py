
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
        #self.columnDimensions = (56, 56)
        self.columnDimensions = (56 * 56)
        self.columnNumber = np.array(self.columnDimensions).prod()
        self.inputSize = np.array(self.inputShape).prod()
        self.spatialPooler = self._initSpatialPooler()

    def _initSpatialPooler(self):

        print "Creating spatial pooler .."

        spatialPooler = SP(
            self.inputSize,   
            self.columnDimensions,
            numActiveColumnsPerInhArea = int(0.02*self.columnNumber),
            globalInhibition = True,
            synPermActiveInc = 0.01,
            potentialPct = 1, # Essential parameter: Neurons can connect to 100% of input
            potentialRadius = self.inputSize
            # stimulusThreshold = 0,
            # synPermInactiveDec = 0.01,
            # synPermActiveInc = 0.1,
            # synPermConnected = 0.1, # Connected threshold
            # maxBoost = 3
        )
        return spatialPooler


    def run(self):
        '''
        Run the digit recognizer 
        '''
        trainingResults = self._train()
        print "Training results: %s" % trainingResults
        testingResults = self._test(trainingResults)
        print "Testing results: %s" % testingResults
        
        for filename, foundMatch in testingResults.items():
            if not foundMatch:
                msg = "Failed to match %s.  Test set: %s" % (filename, self.testingDataDir)
                raise Exception(msg)

        print "All test data in %s was matched!" % self.testingDataDir

    
    def _train(self, numIterations=100):

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

            savedActiveColumns = trainingResults[filename]
            if ((activeColumns == savedActiveColumns).all()):
                testingResults[filename] = True 
            else:
                testingResults[filename] = False


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

    # first run it on the "easy" test set which has very little noise
    _testingDataDirEasy = os.path.join("data", "testing-easy")
    digitRecognizer = DigitRecognizer(_trainingDataDir, _testingDataDirEasy)
    digitRecognizer.run()

    # now run it on the test set that has more noise
    _testingDataDir = os.path.join("data", "testing")
    digitRecognizer = DigitRecognizer(_trainingDataDir, _testingDataDir)
    digitRecognizer.run()
