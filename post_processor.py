import numpy as np
import json as j

import matplotlib.pyplot as plt

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
    def __init__(self, inputMeshPath, outputJsonPath, multip=True, DEBUG=False):
        """The start of the class. Requires the mesh and json file paths to import both.

        Args:
            inputMeshPath (str): The path to the input obj mesh file, before the pre-processing orientation step.
            outputJsonPath (str): The path to the output json file from the model.
            mp (bool, optional): A flag to enable multiprocessing in the minimesh processing step. Defaults to True. The numebr of subprocesses is set to 80% of the cpu count.
        """
        
        # Load the obj mesh file
        self.mesh = t.load(inputMeshPath, process=False)
        
        # Load the json file data
        __json = self._loadJson(outputJsonPath)
        
        self.labels = np.array(__json["labels"]).reshape(-1)
        self.jaw = __json["jaw"]

        self.uniqueTeethLabels = None
        self.centers = None
        self.multiprocess = multip
        self.threadCount = math.floor(mp.cpu_count() * 0.8)
        self.DEBUG = DEBUG

    def _loadJson(self, path):
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
        
        if self.uniqueTeethLabels is not None:
            return self.uniqueTeethLabels
        else:
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
                
                lArg = mp.Value('i', l)
                
                p = mp.Process(target=self.miniMeshCreate, args=(lArg, q), daemon=False, name=str(l)) # Daemon process cannot have children
                _processes.append(p)
            
            for __p in _processes:
                __p.start()
            
            for _p, _q in zip(_processes, _queues):
                
                self.miniMeshes[_p.name] = _q.get()
                _p.join()
                
                _q.close()
                _p.close()
                
                self.centers[_p.name] = self.miniMeshes[_p.name].centroid
            
        else:
            for l in self.getUniqueLabels():
                self.miniMeshes[str(l)] = self.miniMeshCreate(l=l)
                self.centers[str(l)] = self.miniMeshes[str(l)].centroid
        
        
        if self.DEBUG:
            for m in self.miniMeshes:
                self.miniMeshes[m].show()
        
        return self.centers
    
    def miniMeshCreate(self, l, queue=None):
        """A multiprocessable function to create the mini mesh for each tooth label.

        Args:
            l (int): The unique label that we are trying to find.
        """
        
        if queue is not None:
            __label = l.value
        else:
            __label = l
        # Get the indicies of the label
        __idx = self.labels == __label
        __idxValues = np.asarray(self.labels == __label).nonzero()
        
        __t = self.TransposeFaces(__idx)
        
        # Gather the verticies and normals
        __miniMeshV = self.mesh.vertices[__idx]
        
        # Find the faces that are connected to the included verticies
        __miniMeshF = []
        
        self.__faceProcess(faces=self.mesh.faces, idxValues=__idxValues, returnArray=__miniMeshF)
        
        # Translate the faces to the new verticies
        __miniMeshF = __t.transpose(np.array(__miniMeshF))
        

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
        """A class to help transpose face order in the mesh from the origional to ten minimesh. Idealy this can
        be extratcted for other use cases. 
        """ 
        
        def __init__(self, arr):
            """_summary_

            Args:
                arr (np.ndarray): The array of values that represent the indicies that need to be transposed.
            """
            
            self.arr = arr

            __trues = 0
            
            self.transposeArr = []
            for a in self.arr:
                if a:
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
            
            __f = f.copy()
            __f = __f - 1 # f is origionaly 1 indexed
            
            for i, face in enumerate(__f):
                __delete = False
                for j, v in enumerate(face):
                    if self.transposeArr[v] == -1:
                        __delete = True
                        break
                    else:
                        __f[i][j] = self.transposeArr[v]
                if __delete:
                    toDelete.append(i)
                                    
            __f = np.delete(__f, toDelete, axis=0) # Delete the face if it has a -1 (aka one of the verticies is not included)
            
            return __f

class PolynomialWrapper:
    """Wrapper for the numpy.Polynomial so that it works closer to how SKlearn models work.
    """
    
    def __init__(self, degree):
        self.p = None
        self.degree = degree
    
    def fit(self, x, y):
        self.p = Polynomial.fit(x.reshape((-1,)), y.reshape((-1,)), self.degree)
        return self
    
    def predict(self, x):
        """_summary_

        Args:
            x (np.ndarray[1,]): The x value to predict on.
        """
        if not self.p:
            raise ValueError("post_processor::PolynomialWrapper::predict::Error: The model has not been fit yet.")

        return self.p(x)

class JawAlignment:
    """The main class to generate and predict and calculate jaw aligment.
    """
    
    def __init__(self, inputMeshPath, outputJsonPath, multip=True, DEBUG=False, modelEval=False, regressor="", **rKwargs):
        
        self.C = Centeroids(inputMeshPath, outputJsonPath, multip=multip, DEBUG=DEBUG)
        self.meshPath = inputMeshPath
        self.jsonPath = outputJsonPath
        self.mp = multip
        self.debug = DEBUG
        self.centers = self.C.getCenters()
        self.p = None
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.r = (regressor, self.__selectRegressor(regressor, **rKwargs))
        
        if modelEval:
            self.__evaluate()
        
    def __repr__(self):
        return f"JawAlignment of {self.meshPath} and {self.jsonPath} using multiprocessing: {self.mp} with DEBUG set to: {self.debug}"

    def __evaluate(self):
        """A simple function to help evaluate the regressor that is being used. This will calculate the square difference between the predicted and actual points.
        """
        
        print(f"JawAlignment evaluation is on.")
        
        
        print(f"Evaluating regressor: {self.r[0]}")
        print(self.calculateSquareDiff(), self.calculateSquareDiff(mean=True))
        self.plot(r=self.r[0], show=True)

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
            return PolynomialWrapper(2)
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
    
            
        if rDict:
            if points is None:
                c = self.centers

            else:
                c = {k: v[points] for k, v in self.centers.items()}
        
        else:
            if points is None:
                c = np.asarray(list(self.centers.values()))
            
            else:
                c = np.asarray([v[points] for v in self.centers.values()])
            
        
        return c

    def printCenters(self):
        """A function to print the centers of the teeth."""
        
        for l in self.getCenters(rDict=True):
            print(f"Label: {l}, Center: {self.centers[l]}")

    def calculateSquareDiff(self, mean=False):
        """A function to calculate the square difference between predicted and actual points. We will uitilize
        robust regression to find the best fit in light of the included outlier points and then calculate the square
        difference between the predicted and actual points.

        Args:
            mean (bool, optional): A flag to return the mean of the square differences instead of a list of the square differences. Defaults to False.
        Returns:
            d (np.ndarray[n,]): A list of the square differences in the same order as the given points.
        """
        
        c_x = self.getCenters(points=[True, False, False])
        c_y = self.getCenters(points=[False, True, False]).reshape((-1,))
        
        

        if self.p is None:
            if self.r[0].lower() == "base":
                self.p = self.r[1].fit(c_x, c_y)
            else:
                pFeatures = self.poly.fit_transform(c_x.reshape((-1, 1)))
                self.p = self.r[1].fit(pFeatures, c_y)
        
        d = np.empty((c_y.shape[0],))
        
        
        if self.r[0] == "base": 
            for i, (x, y) in enumerate(zip(c_x, c_y)):
                d[i] = (self.p.predict(x) - np.asarray(y))**2
        else:
            for i, (x, y) in enumerate(zip(c_x, c_y)):
                d[i] = (self.p.predict(self.poly.transform(np.asarray([x]).reshape((1, 1)))) - np.asarray(y))**2
        
        if mean:
            return np.asarray(np.mean(d))
        else:
            return d

    def plot(self, r="", save=False, savePath=None, show=True):
        """A function to plot the predicted and actual points.
        
        Args:
            save (bool, optional): A flag to save the plot. Defaults to False.
            savePath (str, optional): The path to save the plot to. Defaults to None.
            show (bool, optional): A flag to show the plot. Defaults to True.
        """
        
        c_x = self.getCenters(points=[True, False, False]).reshape((-1,))
        c_y = self.getCenters(points=[False, True, False]).reshape((-1,))
        
        c_x, c_y = zip(*sorted(zip(c_x, c_y)))
        
        fig, ax = plt.subplots()
    
        fig.tight_layout()
        ax.set_title("Jaw Alignment Prediction Alignment Augmented Results")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.scatter(c_x, c_y, label="Actual Teeth Centers") # The calculated centers of the teeth as a scatter plot     
        
        if self.p is None:
            _ = self.calculateSquareDiff()
        
        if self.r[0] == "base":
            ax.plot(c_x, self.p.predict(np.array(c_x)), label=f"Predicted Teeth Centers ({r} Regressor)", color="red")
        else:
            ax.plot(c_x, self.p.predict(self.poly.transform(np.array(c_x).reshape((-1, 1)))), label=f"Predicted Teeth Centers ({r} Regressor)", color="green")
        
        
        ax.legend()
        
        if save:
            if savePath is None:
                raise ValueError("post_processor::JawAlignment::plot::Error: No save path given.")
            
            fig.savefig(savePath)
        
        if show:
            plt.show()
        

def main(argv):
    
    __rKargs = { "base" : {},
                "ransac": {"min_samples": 2, "max_trials": 10000, "loss": "absolute_error"},
                "huber": {"epsilon": 1.15, "max_iter": 10000, "alpha": 0.0001}         
        }
    
    __TEST = "huber"
    
    C = JawAlignment(argv.input_mesh, argv.output_json, multip=True, DEBUG=False, modelEval=True, regressor=__TEST, **__rKargs[__TEST])


if __name__ == '__main__':
    
    parser = ap.ArgumentParser(description="Post processing steps and helpful tools on input mesh and output json labels.")
    parser.add_argument('-i', '--input-mesh', type=str, help='Path to the input mesh obj file. Make sure that this is the file before the pre-processing orientation step.')
    parser.add_argument('-j', '--output-json', type=str, help='Path to the output json file from the model.')
    
    argv = parser.parse_args()
    
    main(argv)