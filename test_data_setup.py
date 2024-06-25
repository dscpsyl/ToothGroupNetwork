import trimesh as t
import math
import argparse as ap



def stl2obj(stl_file, obj_file):
    t.load(stl_file).export(obj_file, digits=15, include_normals=True)


def upperProcess(mesh):
    center = [0, 0, 0]
    
    angleX = math.pi / 2
    directionX = [1, 0, 0]
    
    rotX = t.transformations.rotation_matrix(angleX, directionX, center)
    mesh.apply_transform(rotX)
    
    angleY = math.pi
    directionY = [0, 1, 0]
    
    rotY = t.transformations.rotation_matrix(angleY, directionY, center)
    mesh.apply_transform(rotY)
    
    return mesh

def lowerProcess(mesh):
    center = [0, 0, 0]
    
    angleX = math.pi / 2
    directionX = [1, 0, 0]
    
    rotX = t.transformations.rotation_matrix(angleX, directionX, center)
    mesh.apply_transform(rotX)
    
    return mesh

def main(args):
    
    if args.stl_file is None or args.obj_file is None:
        print('Please provide input and output file')
        return

    if not args.stl_file.lower().endswith('.stl') or not args.obj_file.lower().endswith('.obj'):
        print('Please provide a valid input and output file')
        return
    
    if args.part not in ['u', 'l']:
        print('Please provide a valid part')
        return
    
    mesh = t.load(args.stl_file)
    
    if args.part == 'u':
        mesh = upperProcess(mesh)
    else:
        mesh = lowerProcess(mesh)
    
    mesh.export(args.obj_file, digits=15, include_normals=True)

if __name__ == '__main__':
    parser = ap.ArgumentParser(description='Test data preprocessing to get it ready for the model')
    parser.add_argument('-s', '--stl-file', type=str, help='Input stl file')
    parser.add_argument('-o', '--obj-file', type=str, help='Output obj file')
    parser.add_argument('-p', '--part', type=str, help= "Either 'u' for upper part or 'l' for lower part")
    args = parser.parse_args()

    main(args)