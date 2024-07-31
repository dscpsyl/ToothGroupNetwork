import torch.cuda
from torchvision import transforms as T
import torch
import cv2 as c
import nibabel as nib
import numpy as np

from PIL import Image

import argparse as ap

class resizeCv2(object):
    def __init__(self, size, save=False, savePath="", show=False):
        """Resize an image with CV2. This is basically a wrapper for the cv2.resize function. It is written as a class so that it can be used in a pipeline.

        Args:
            imgPath (str): The path to the image.
            size (tuple): The size of the image. If the values are between 0 and 1, then it is considered as a fraction of the original
            save (bool, optional): Whether to save the resulting image or not. Defaults to False.
            savePath (str, optional): Where to save the resulting image. Defaults to "" but required if @param save is true..
            show (bool, optional): Whether to show the image or not. Defaults to False.

        Raises:
            ValueError: Raised if @param save is true but @param savePath is empty.

        Returns:
            cv2.image: The new resized image.
        """
        
        self.size = size
        self.save = save
        self.savePath = savePath
        self.show = show
    
    def __call__(self, img):
        # Check if we are resizing based on absolute or relative
        if 0 < self.size[0] < 1 and 0 < self.size[1] < 1:
            d = c.resize(img, (0, 0), fx=self.size[0], fy=self.size[1])
        else:
            d = c.resize(img, self.size)
        
        if self.save:
            if self.savePath == "":
                raise ValueError("resizeWithCv2::savePath is empty")
            
            c.imwrite(self.savePath, d)
        
        if self.show:
            c.imshow("Resized Image", d)
            c.waitKey(0)
            c.destroyAllWindows()
    
        return d
    
    def __repr__(self):
        return self.__class__.__name__+'()'

class resizePIL(object):
        def __init__(self, size, save=False, savePath="", filter=None):            
            self.size = size
            self.save = save
            self.savePath = savePath
            self.filter = filter
    
        def __call__(self, img):
            
            d = img.resize(self.size, self.filter)
            
            
            if self.save:
                if self.savePath == "":
                    raise ValueError("resizeWithCv2::savePath is empty")
                
                c.imwrite(self.savePath, d)
        
            return d
        
        def __repr__(self):
            return self.__class__.__name__+'()'

class cvtColorCv2(object):
    """A wrapper for the cv2.cvtColor function. This is written as a class so that it can be used in a pipeline.
    """
    
    def __init__(self, codeword):
        """Initialize the class with the codeword for the conversion.

        Args:
            codeword (int): The codeword for the conversion.
        """
        
        self.codeword = codeword
        
    
    def __call__(self, img):
        return c.cvtColor(img, self.codeword)

    def __repr__(self):
        return self.__class__.__name__+'()'

class assertShapePipelineCv2(object):
    """A class wrapper for the assert function to check the size output of a pipeline step.
    """
    
    def __init__(self, assertion):
        """Initialize the class with the assertion to be made.

        Args:
            assertion (any): The assertion of the size to be made.
        """
        
        self.assertion = assertion
    
    def __call__(self, img):
        assert img.shape == self.assertion, f"assertPipeline::Expected {self.assertion} but got {img.size}"
        return img

    def __repr__(self):
        return self.__class__.__name__+ f'({self.assertion})'

class assertShapePipelinePIL(object):
    """A class wrapper for the assert function to check the size output of a pipeline step.
    """
    
    def __init__(self, assertion):
        """Initialize the class with the assertion to be made.

        Args:
            assertion (any): The assertion of the size to be made.
        """
        
        self.assertion = assertion
    
    def __call__(self, img):
        assert img.size == self.assertion, f"assertPipeline::Expected {self.assertion} but got {img.size}"
        return img

    def __repr__(self):
        return self.__class__.__name__+ f'({self.assertion})'
    

def fractionalPositionCv2(img, point):
    """Find the fractional position of a point in an image. Point (0,0) is the top left corner of the image, This fraction is the porportional distance from
    (0, 0) to the point in the image.

    Args:
        img (cv2.img): A loaded cv2 image.
        point (tuple): A tuple containing the (x, y) position of the point. The point is 0-indexed.

    Raises:
        ValueError: Raised if @param point is not within the bounds of the image.
    Returns:
        tuple: A tuple of the fractional (x, y) position of the point in the image.
    """
    
    __h, __w, _ = img.shape
    
    _x = point[0]
    _y = point[1]
    
    if _x < 0 or _y < 0:
        raise ValueError("fractionalPosition::Point is out of bounds")
    if _x > __w or _y > __h:
        raise ValueError("fractionalPosition::Point is out of bounds")
    
    fX = _x / __w
    fY = _y / __h
    
    return (fX, fY)

def _preprocessScanCV2(pipelines, IPATH, OPATH):
    # Load the iamge
    img = c.imread(IPATH)
    
    # Apply the transformations
    for pipeline in pipelines:
        img = pipeline(img)
    
    # Normalize the image and convert it to a tensor
    m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=m, std=s),
    ])
    
    inputTensor = preprocess(img)
    inputTensor = inputTensor.unsqueeze(0)
    
    # Check if CUDA is available on the divice and transfer it over
    if torch.cuda.is_available():
        inputTensor = inputTensor.cuda()
    
    #Save the tensor
    torch.save(inputTensor, OPATH + ".pt")  

def _preprocessGroundTruthCV2(pipelines, MPATH, OPATH, type="npy"):
    
    if type == "nii": 
        # Load the mask with the mask specific transformations
        mask = nib.load(MPATH) # There might be multiple channels but they should all be the same since this is a mask.
        mask = mask.get_fdata()
        mask = np.float32(mask)
        
        # If there are more than three channels in the mask, then we need to remove the extra channels
        if mask.shape[2] > 3:
            newMask = mask[:, :, :3]
        
        # If there are less than three channels, then we need to add the extra channels
        if mask.shape[2] < 3:
            layer = mask[:, :, 0]
            newMask = np.stack((layer, layer, layer), axis=2)
        
        assert newMask.shape == (mask.shape[0], mask.shape[1], 3), f"MRI.preprocess::main::Expected {(mask.shape[0], mask.shape[1], 3)} but got {newMask.shape}"
        mask = newMask
    
    else:
        mask = np.load(MPATH)
        mask = np.float32(mask)
        newMask = np.stack((mask, mask, mask), axis=2)
        
        assert newMask.shape == (mask.shape[0], mask.shape[1], 3), f"MRI.preprocess::main::Expected {(mask.shape[0], mask.shape[1], 3)} but got {newMask.shape}"
        mask = newMask
    
    # Apply the transformations
    for pipeline in pipelines:
        mask = pipeline(mask)
    
    c.imshow(f"Pred Mat {MPATH}", mask)
    c.waitKey(0)
    c.destroyAllWindows()

    
    # Save the mask as a tensor
    tMask = torch.from_numpy(mask)
    tMask = tMask.unsqueeze(0)
    
    if torch.cuda.is_available():
        tMask = tMask.cuda()
    
    torch.save(tMask, OPATH + "-gt.pt")
    
def mainCv2(args):
    
    # Create the preprocess pipeline
    preprocessAllCv2 = T.Compose([
        # Resize and recolorcode the image
        resizeCv2((256, 256)),
        cvtColorCv2(c.COLOR_BGR2RGB), # REMEMBER THAT CV2 LOADS THE IMAGE IN BGR FORMAT INSTEAD OF RGB
        
        # Assert the transformations are correct
        assertShapePipelineCv2((256, 256, 3))
    ])
    
    # Pass it into the preprocess functions as a list in case we need to add more subpipelines in the future
    _preprocessScanCV2([preprocessAllCv2], args.mri_scan, args.output)
    _preprocessGroundTruthCV2([preprocessAllCv2], args.mask, args.output, type=args.type)

def _preprocessScanPIL(pipelines, IPATH, OPATH):
    # Load the iamge
    img = Image.open(IPATH)
    
    # Apply the transformations
    for pipeline in pipelines:
        img = pipeline(img)
    
    # Normalize the image and convert it to a tensor
    m, s = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
    print(m, s)
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=m, std=s),
    ])
    
    inputTensor = preprocess(img)
    inputTensor = inputTensor.unsqueeze(0)
    
    # Check if CUDA is available on the divice and transfer it over
    if torch.cuda.is_available():
        inputTensor = inputTensor.cuda()
    
    #Save the tensor
    torch.save(inputTensor, OPATH + ".pt")  

def _preprocessGroundTruthPIL(pipelines, MPATH, OPATH, type="npy"):
    if type == "nii": 
        # Load the mask with the mask specific transformations
        mask = nib.load(MPATH) # There might be multiple channels but they should all be the same since this is a mask.
        mask = mask.get_fdata()
        mask = np.float32(mask)
        
        # If there are more than three channels in the mask, then we need to remove the extra channels
        if mask.shape[2] > 3:
            newMask = mask[:, :, :3]
        
        # If there are less than three channels, then we need to add the extra channels
        if mask.shape[2] < 3:
            layer = mask[:, :, 0]
            newMask = np.stack((layer, layer, layer), axis=2)
        
        assert newMask.shape == (mask.shape[0], mask.shape[1], 3), f"MRI.preprocess::main::Expected {(mask.shape[0], mask.shape[1], 3)} but got {newMask.shape}"
        mask = newMask
    
    else:
        mask = np.load(MPATH)
        mask = np.float32(mask)
        newMask = np.stack((mask, mask, mask), axis=2)
        
        assert newMask.shape == (mask.shape[0], mask.shape[1], 3), f"MRI.preprocess::main::Expected {(mask.shape[0], mask.shape[1], 3)} but got {newMask.shape}"
        mask = newMask
    
    mask = Image.fromarray(np.uint8(mask))

    # Apply the transformations
    for pipeline in pipelines:
        mask = pipeline(mask)
    
    mask.point(lambda p: 1 if p > 0.5 else 0)
    
    mask.show()
    
    # Save the mask as a tensor
    tMask = torch.from_numpy(mask)
    tMask = tMask.unsqueeze(0)
    
    if torch.cuda.is_available():
        tMask = tMask.cuda()
    
    torch.save(tMask, OPATH + "-gt.pt")

def mainPIL(args):
    
    preprocessAllPIL = T.Compose([
      resizePIL((256, 256)),
      
      assertShapePipelinePIL((256, 256))
    
    ])
    
    _preprocessScanPIL([preprocessAllPIL], args.mri_scan, args.output)
    _preprocessGroundTruthPIL([preprocessAllPIL], args.mask, args.output, type=args.type)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Preprocess the MRI data for the U-Net model")
    parser.add_argument("-s", "--mri-scan", help="The path to the MRI scan image", required=True, type=str)
    parser.add_argument("-m", "--mask", help="The path to the mask image", required=True, type=str)
    parser.add_argument("-t", "--type", help="The type of the mask image", default="npy", type=str, choices=["npy", "nii"])
    parser.add_argument("-o", "--output", help="The output path for the preprocessed images", required=True, type=str)
    
    args = parser.parse_args()
    
    # mainCv2(args)
    mainPIL(args)