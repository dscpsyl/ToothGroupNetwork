import numpy as np
import json as j

import matplotlib.pyplot as plt
import os.path as op

import trimesh as t
from numpy.polynomial import Polynomial

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures

import argparse as ap
import multiprocessing as mp
import math

class Centeroids:
    """A class to help organize the process in finding the average point location of each teeth in the mesh.
    """
    def __init__(self, inputMeshPath, outputJsonPath, multiProcess=True, debug=False, mpCount=0.9):
        """The start of the class. Requires the mesh and json file paths to import both.

        Args:
            inputMeshPath (str): The path to the input obj mesh file, before the pre-processing orientation step.
            outputJsonPath (str): The path to the output json file from the model.
            mp (bool, optional): A flag to enable multiprocessing in the minimesh processing step. Defaults to True.
            debug (bool, optional): A flag to show the mini meshes. Defaults to False.
            mpCount (float, optional): The percentage of the cpu count to use for multiprocessing. Defaults to 0.9.
        """
        
        # Load the obj mesh file
        self.mesh = t.load(inputMeshPath, process=False)
        
        # Load the json file data
        __json = self.loadJson(outputJsonPath)
        
        self.labels = np.array(__json["labels"]).reshape(-1)
        self.jaw = __json["jaw"]

        self.uniqueTeethLabels = None
        self.centers = None
        self.miniMeshes = None
        
        self.multiprocess = multiProcess
        self.threadCount = math.floor(mp.cpu_count() * mpCount)
        self.debug = debug

    def loadJson(self, path):
        """A helper function that mirrors render.py:_load_json function
        in reading and processing the data from a json file.
        
        Args:
            path (str): The path to the json file.
            
        Returns:
            js (dict): The json data as a dictionary.
        """
        
        with open(path, 'r') as f:
            js = j.load(f)
        
        return js
        
    def getUniqueLabels(self):
        """A function to return the unique labels from the json data, excluding zero, which is gum and we don't 
        care about that. Will not recalculate if the unique labels have already been calculated.
        
        Returns:
            uniqueLabels (np.array): The unique labels from the json data.
        """
        
        if self.uniqueTeethLabels is None:
            self.uniqueTeethLabels = np.unique(self.labels)
            self.uniqueTeethLabels = self.uniqueTeethLabels[np.nonzero(self.uniqueTeethLabels)]
    
        return self.uniqueTeethLabels
        
    def calculate(self):
        """A function to calculate all the average center points for each of the segmented teeth.
        It uitilizes self.meshLS for the verticies and norms to create mini meshes and uitilizes trimesh
        to figure out the centeroid of each tooth.
        
        Returns:
            centers (dict): A dictionary of the centeroid points for each tooth label in the form of {str(label):[3d point]}.
        """
        
        self.centers = {}
        self.miniMeshes = {}
        
        if self.multiprocess:
            if self.threadCount < len(self.getUniqueLabels()):
                input("post_processor::Centeroids::calculate::Warning: The number of threads is less than the number of unique labels. This may cause the program to run slower and your computer to freeze up while the calculations are being completed. Press enter to continue.")
                
            _processes = []
            _queues = []
            
            for l in self.getUniqueLabels():
                q = mp.Queue()
                _queues.append(q)
                
                p = mp.Process(target=self.miniMeshCreate, args=(l, q), daemon=False, name=str(l)) # Daemon process cannot have children
                _processes.append(p)
                
                p.start()
            
            for _p, _q in zip(_processes, _queues):
                
                self.miniMeshes[_p.name] = _q.get()
                
                _p.join()
                
                self.centers[_p.name] = self.miniMeshes[_p.name].centroid
            
        else:
            for l in self.getUniqueLabels():
                self.miniMeshes[str(l)] = self.miniMeshCreate(l=l)
                self.centers[str(l)] = self.miniMeshes[str(l)].centroid
        
        
        if self.debug:
            for m in self.miniMeshes:
                self.miniMeshes[m].show()
        
        return self.centers
    
    def miniMeshCreate(self, l, queue=None):
        """A multiprocessable function to create the mini mesh for each tooth label.

        Args:
            l (int): The unique label that we are trying to find.
        """
        
        # Get the indicies of the label
        __idx = self.labels == l
        __idxValues = __idx.nonzero()
        
        __t = self.TransposeFaces(__idx)
        
        # Gather the verticies and normals
        __miniMeshV = self.mesh.vertices[__idx]
        
        # Find the faces that are connected to the included verticies
        __miniMeshF = []
        self.__faceProcess(faces=self.mesh.faces, idxValues=__idxValues, returnArray=__miniMeshF)
        
        # Translate the faces to the new verticies
        __miniMeshF = __t.transpose(np.asarray(__miniMeshF))
        
        __miniMesh = t.Trimesh(vertices=__miniMeshV, vertex_normals=self.mesh.vertex_normals, faces=__miniMeshF, process=False)
        
        if queue is not None:
            queue.put(__miniMesh)
        else:
            return __miniMesh
    
    def __faceProcess(self, queue=None, faces=None, idxValues=None, returnArray=None):
        """A multiprocessable helper function to process the faces of the mini mesh.

        Args:
            idxValues (np.ndarray): The array of idxValues to compare to.
            returnArray (np.ndarray): The resulting array to append to.
        """
        
        if queue is not None:
            faces, idxValues = queue.get()
            returnArray = []
        
        for f in faces:
            if np.any(np.isin(f - 1, idxValues)): # Minus 1 because the faces are 1 indexed
                returnArray.append(f)
        
        if queue is not None:
            queue.put(np.asarray(returnArray))
    
    def getCenters(self):
        """A function to return the centers of the different teeth labels. Will calculate if not already
        done so.
        
        Returns:
            centers (dict): A dictionary of the centeroid points for each tooth label in the form of {label:[3d point]}.
        """
        
        if self.centers is None:
            self.calculate()
        
        return self.centers
            
    class TransposeFaces:
        """A class to help transpose face order in the mesh from the origional to the minimesh. Idealy this can
        be extratcted for other use cases. 
        """ 
        
        def __init__(self, arr):
             
            self.transposeArr = []
            __trues = 0
            
            for a in arr:
                if a: # if it is anything but 0
                    self.transposeArr.append(__trues)
                    __trues += 1
                else:
                    self.transposeArr.append(-1)
        
        def transpose(self, f):
            """The function that takes in a faces array and transposes the elements based on the transpose array calculated with the init function.
            This will also delete any face that does not have all its verticies as a curtsey.

            Args:
                f (np.ndarray [nx3]): A faces array from Trimesh that needs to be transposed. 

            Returns:
                t (np.ndarray [nx3]): The transposed faces array.
            """
            
            toDelete = []
            
            fZero = f - 1 # f is origionaly 1 indexed
            
            for i, face in enumerate(fZero):
                __delete = False
                
                for j, v in enumerate(face):
                    if self.transposeArr[v] == -1:
                        __delete = True
                        break
                    else:
                        fZero[i][j] = self.transposeArr[v]
                
                if __delete:
                    toDelete.append(i)
                                    
            fR = np.delete(fZero, toDelete, axis=0) # Delete the face if it has a -1 (aka one of the verticies is not included)
            
            return fR

class PolynomialWrapper:
    """Wrapper for the numpy.Polynomial so that it works closer to how SKlearn models work.
    """
    
    def __init__(self, degree):
        
        self.p = None
        self.degree = degree
    
    def fit(self, x, y):
        """Just to wrap the Polynomial.fit function.

        Args:
            x (np.ndarray): The x values
            y (np.ndarray): They y values

        Returns:
            self (self): This class instance again with the model fitted.
        """
        
        _x = x.reshape((-1,))
        _y = y.reshape((-1,))
        
        if len(_x) != len(_y):
            raise ValueError("post_processor::PolynomialWrapper::fit::Error: The x and y values are not the same length.")
        
        self.p = Polynomial.fit(_x, _y, self.degree)
        
        return self
    
    def predict(self, x):
        """Just to wrap the Polynomial.__call__ function.

        Args:
            x (np.ndarray): The x value to predict on.
        """
        
        _x = x.reshape((-1,))
        
        if not self.p:
            raise ValueError("post_processor::PolynomialWrapper::predict::Error: The model has not been fit yet.")

        y = self.p(_x)

        return np.array(y)

class JawAlignment:
    """The main class to generate and predict and calculate jaw aligment.
    """
    
    def __init__(self, inputMeshPath, outputJsonPath, multiProcess=True, debug=False, modelEval=False, regressor="", **rKwargs):
        
        self.C = Centeroids(inputMeshPath, outputJsonPath, multiProcess=multiProcess, debug=debug)
        self.meshPath = inputMeshPath
        self.jsonPath = outputJsonPath
        self.mp = multiProcess
        self.debug = debug
        self.degree = rKwargs["degree"]
        
        self.centers = self.C.getCenters()
        
        self.p = None
        self.polyFilter = [1, 3, 5] #blacklist
        self.poly = PolynomialFeatures(degree=rKwargs["degree"]) # We include bias just for easier reasoning of which columns to delete
        _ = rKwargs.pop("degree", None)
        self.r = (regressor, self.__selectRegressor(regressor, **rKwargs))
        
        if modelEval:
            self.__evaluate()
        
    def __repr__(self):
        return f"JawAlignment of '{self.meshPath}' and '{self.jsonPath}' using multiprocessing: '{self.mp}' with debug set to: '{self.debug}' and regressor: '{self.r[0]}'"
    
    def __evaluate(self):
        """A simple function to help evaluate the regressor that is being used. This will calculate the square difference between the predicted and actual points.
        """
        
        print(f"JawAlignment evaluation is on.")
        print(f"Evaluating: {self}")
        
        d, m = self.calculateSquareDiff(mean=True)
        print(f"Square Differences: {d}")
        print(f"Mean Square Difference: {m}")
        
        self.plot(show=True)

    def __selectRegressor(self, regressor, **kwargs):
        """A helper function to select which regressor to use for this class.

        Args:
            regressor (str): The regressor for the model to fit with.
            **kwargs: Additional arguments to pass to the regressor.
        
        Returns:
            r (None | sklearn.linear_model): The regressor to use.
        """
        
        if regressor.lower() == "huber":
            return HuberRegressor(**kwargs)
        elif regressor.lower() == "base":
            return PolynomialWrapper(self.degree)
        elif regressor.lower() == "ransac":
            return RANSACRegressor(**kwargs)
        else:
            raise ValueError("post_processor::JawAlignment::__selectRegressor::Error: Invalid regressor type.")

    def getCenters(self, rDict=False, points=None):
        """Grabs the centeroid information from the Centeroids class. Can return simply the raw center datapoints or
        a dictionary of the center points with its corresponding label.

        Args:
            dict (bool, optional): Whether to return as an array or a dict. Defaults to False.
            points (np.ndarray[n,3], optional): The axis to return [x, y, z]. If none, will return all. Defaults to None.

        Returns:
            c (np.ndarray[n,3] | dict["label":np.ndarray]): The center points of the teeth.
        """
    
            
        if rDict: # If we need to output as dict
            
            if not points: # If we need all points 
                c = self.centers
            else: # If we need only certain points
                c = {k: v[points] for k, v in self.centers.items()}
        
        else: # If we need to output as list
            
            if not points: # If we need all points
                c = np.asarray(list(self.centers.values()))
            else: # If we need only certain points
                c = np.asarray([v[points] for v in self.centers.values()])
            
        return c

    def printCenters(self):
        """A function to print the centers of the teeth."""
        
        for l, c in self.getCenters(rDict=True).items():
            print(f"Label: {l}, Center: {c}")

    def translateAboveZero(self, x, y, sort=True):
        """A helper function to translate all coordinate points above zero. The points will also be sorted in ascending order.

        Args:
            x (np.ndarray): The x points.
            y (np.ndarray[n,]): The y points.
            sorted (bool, optional): A flag to sort the points. Defaults to True.
            
        Returns
            p (tuple(np.ndarray(x), np.ndarray(y))): A typle of x and y points translated above zero, with optional sorting.
        """
        
        if sort:
            _x, _y = zip(*sorted(zip(np.array(x).reshape((-1,)), np.array(y).reshape((-1,)))))

        lowest = np.min(_y) 
        
        if lowest < 0:
            _y = _y - lowest # since lowest is a negative number
        
        return np.array(_x), np.array(_y)

    def __getPreprocessedPoints(self, raw=False):
        """A helper function to return pre-processed points that the regressors can work with. Should not be called outside of the class.
        
        Args:
            raw (bool, optional): A flag to return the raw x points. Defaults to False.
        
        Returns:
            p (tuple(np.ndarray[n,], np.ndarray[n,])): The preprocessed points.
        """

        __x = self.getCenters(points=[True, False, False])
        __y = self.getCenters(points=[False, True, False]).reshape((-1,))    

        __x, y = self.translateAboveZero(__x, __y)
        
        if self.r[0].lower() == "base":
            x = __x
        else:
            x = self.poly.fit_transform(__x.reshape((-1, 1)))
            x = np.delete(x, self.polyFilter, axis=1) # We delete the columns that we don't need (everything bot x^2 and x^6 and x^0)
        
        if raw:
            return __x, x, y
        else:
            return x, y

    def calculateSquareDiff(self, mean=False):
        """A function to calculate the square difference between predicted and actual points. We will uitilize
        robust regression to find the best fit in light of the included outlier points and then calculate the square
        difference between the predicted and actual points.

        Args:
            mean (bool, optional): A flag to return a tuple that inlcudes the mean as the second entry. Defaults to False.
        Returns:
            d ( np.ndarray[n,] | tuple(np.ndarray[n,], mean)): A list of the square differences in the same order as the given points. If mean is True, will return a tuple with the mean as the second entry.
        """

        pX, c_y = self.__getPreprocessedPoints()
        
        if self.p is None:    
            self.p = self.r[1].fit(pX, c_y)
        
        d = np.empty((c_y.shape[0],))
        for i, (x, y) in enumerate(zip(pX, c_y)):
            d[i] = (self.p.predict(x.reshape((1, -1))) - np.asarray(y))**2
        
        if mean:
            return np.array(d), np.mean(d)
        else:
            return d

    def plot(self, save=False, savePath="", show=True):
        """A function to plot the predicted and actual points.
        
        Args:
            save (bool, optional): A flag to save the plot. Defaults to False.
            savePath (str, optional): The path to save the plot to. Defaults to None.
            show (bool, optional): A flag to show the plot. Defaults to True.
        """
        
        c_x, pX, c_y = self.__getPreprocessedPoints(raw=True)
        
        fig, ax = plt.subplots()
    
        fig.tight_layout()
        ax.set_title("Jaw Alignment Prediction Alignment Augmented Results")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.scatter(c_x, c_y, label="Actual Teeth Centers", color="blue") # The calculated centers of the teeth as a scatter plot     
        
        _, m = self.calculateSquareDiff(mean=True)
        __text = f"Patient: {op.basename(self.meshPath).split('_')[0]} \nMean Square Difference: {m:.2f}"
        ax.text(0.5, 0.5, __text, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
        ax.plot(c_x, self.p.predict(pX), label=f"Predicted Teeth Centers ({self.r[0]} regressor)", color="red")
        
        ax.legend()
        
        if save:
            if savePath is None:
                raise ValueError("post_processor::JawAlignment::plot::Error: No save path given.")
            
            fig.savefig(savePath)
        
        if show:
            plt.show()
        
        plt.close(fig)
        

def main(argv):
    
    __rKargs = { "base" : {"degree": 6},
                "ransac": {"degree": 6, "min_samples": 2, "max_trials": 10_000, "loss": "absolute_error"},
                "huber": {"degree" : 6, "epsilon": 1.15, "max_iter": 100_000, "alpha": 0.000_1, "fit_intercept" : False}         
        }
    
    __TEST = argv.regressor
    
    _ = JawAlignment(argv.input_mesh, argv.output_json, multiProcess=True, debug=False, modelEval=True, regressor=__TEST, **__rKargs[__TEST])


def BATCHSAMPLE():
    """TMP::DEBUG
    """
    SEED = 69
    SAMPLES = 10
    
    __rKargs = { "base" : {"degree": 6},
                "ransac": {"degree": 6, "min_samples": 2, "max_trials": 10_000, "loss": "absolute_error"},
                "huber": {"degree" : 6, "epsilon": 1.15, "max_iter": 100_000, "alpha": 0.000_1, "fit_intercept" : False}         
        }
    
    import random
    import os
    
    random.seed(SEED) # make consistent
    
    with open("../data/base_name_test_fold.txt") as f:
        datapoints = f.readlines()
    
    arch = random.sample(datapoints, SAMPLES)
    
    sDir = "../data/post_processing_results"
    
    for point in arch:
        
        s = op.basename(point).strip().split('_')[0]
        os.makedirs(f"{sDir}/{s}", exist_ok=True)
        
        for jaw in ["upper", "lower"]:
            
            mP = f"/home/user/Downloads/3DToothSegmentation/data/data_obj_parent_directory/{s}/{s}_{jaw}.obj"
            jP = f"/home/user/Downloads/3DToothSegmentation/testing_results/{s}_{jaw}.json"
            
            for regressor in ["base", "ransac", "huber"]:
                
                sdj = JawAlignment(mP, jP, multiProcess=True, debug=False, modelEval=False, regressor=regressor, **__rKargs[regressor])
                sdj.plot(save=True, savePath=f"{sDir}/{s}/{s}_{jaw}_{regressor}.png", show=False)
                
                print(f"Finished: {s}_{jaw}_{regressor}")
    

if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description="Post processing steps and helpful tools on input mesh and output json labels.")
    parser.add_argument('-i', '--input-mesh', type=str, help='Path to the input mesh obj file. Make sure that this is the file before the pre-processing orientation step.')
    parser.add_argument('-j', '--output-json', type=str, help='Path to the output json file from the model.')
    parser.add_argument('-r', '--regressor', type=str, choices=["base", "ransac", "huber"], help="The regressor to use for the model. Can be 'base', 'ransac', or 'huber'.", default="huber")
    
    argv = parser.parse_args()
    
    # main(argv)
    
    BATCHSAMPLE()