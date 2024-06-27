import sys
import os
import trimesh
sys.path.append(os.getcwd())
import numpy as np
import open3d as o3d
import argparse
import json

parser = argparse.ArgumentParser(description='Inference models')
parser.add_argument('--mesh_path', default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files/013FHA7K/013FHA7K_lower.obj", type=str)
# parser.add_argument('--gt_json_path', default="G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances/013FHA7K/013FHA7K_lower.json" ,type=str)
parser.add_argument('--pred_json_path', type=str, default="test_results/013FHA7K_lower.json")
args = parser.parse_args()


def _load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)
    
def _get_colored_mesh(mesh, label_arr):
    palte = np.array([
		[255,153,153],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50],
        
        [56, 93, 56],
        [87, 51, 40],
        [0, 59, 81],
        [43, 72, 82],
        [69, 48, 40],
        [95, 84, 57],
        [50, 68, 50]

    ])/255
    # palte[9:] *= 0.4
    label_arr = label_arr.copy()
    label_arr %= palte.shape[0]
    label_colors = np.zeros((label_arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[label_arr==idx] = palte[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
    print(np.asarray(mesh.vertex_colors))
    return mesh
    
def _read_txt_obj_ls(path, ret_mesh=False, use_tri_mesh=False):
    # In some cases, trimesh can change vertex order
    if use_tri_mesh:
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces)+1
    else:
        f = open(path, 'r')
        vertex_ls = []
        tri_ls = []
        #vertex_color_ls = []
        while True:
            line = f.readline().split()
            if not line: break
            if line[0]=='v':
                vertex_ls.append(list(map(float,line[1:4])))
                #vertex_color_ls.append(list(map(float,line[4:7])))
            elif line[0]=='f':
                tri_verts_idxes = list(map(str,line[1:4]))
                if "//" in tri_verts_idxes[0]:
                    for i in range(len(tri_verts_idxes)):
                        tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
                tri_verts_idxes = list(map(int, tri_verts_idxes))
                tri_ls.append(tri_verts_idxes)
            else:
                continue
        f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls)-1)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)

    vertex_ls = np.array(vertex_ls)
    output = [np.concatenate([vertex_ls,norms], axis=1)]

    if ret_mesh:
        output.append(mesh)
    return output

def _print_3d(*data_3d_ls):
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd(item)
    o3d.visualization.draw_geometries(data_3d_ls, mesh_show_wireframe = False, mesh_show_back_face = True)

# def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
#     ins_label_names = np.unique(pred_ins_labels)
#     ins_label_names = ins_label_names[ins_label_names != 0]
#     IOU = 0
#     F1 = 0
#     ACC = 0
#     SEM_ACC = 0
#     IOU_arr = []
#     for ins_label_name in ins_label_names:
#         #instance iou
#         ins_label_name = int(ins_label_name)
#         ins_mask = pred_ins_labels==ins_label_name
#         gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
#         gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
#         gt_mask = gt_labels == gt_label_name

#         TP = np.count_nonzero(gt_mask * ins_mask)
#         FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
#         FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
#         TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

#         ACC += (TP + TN) / (FP + TP + FN + TN)
#         precision = TP / (TP+FP)
#         recall = TP / (TP+FN)
#         F1 += 2*(precision*recall) / (precision + recall)
#         IOU += TP / (FP+TP+FN)
#         IOU_arr.append(TP / (FP+TP+FN))
#         #segmentation accuracy
#         pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
#         sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
#         if is_half:
#             if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
#                 SEM_ACC +=1
#         else:
#             if sem_label_name == gt_label_name:
#                 SEM_ACC +=1
#         #print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
#     return IOU/len(ins_label_names), F1/len(ins_label_names), ACC/len(ins_label_names), SEM_ACC/len(ins_label_names), IOU_arr

# gt_loaded_json = _load_json(args.gt_json_path)
# gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)

pred_loaded_json = _load_json(args.pred_json_path)
pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

# IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels) # F1 -> TSA, SEM_ACC -> TIR
# print("IoU", IoU, "F1(TSA)", F1, "SEM_ACC(TIR)", SEM_ACC)
_, mesh = _read_txt_obj_ls(args.mesh_path, ret_mesh=True, use_tri_mesh=True)

_print_3d(_get_colored_mesh(mesh, pred_labels)) # color is random
# _print_3d(_get_colored_mesh(mesh, gt_labels)) # color is random
