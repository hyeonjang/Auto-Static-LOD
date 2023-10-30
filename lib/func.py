import re, io
from pathlib import Path
import imageio

import torch
import numpy as np

def to_numpy(*args):
    def convert(a):
        if isinstance(a,torch.Tensor):
            return a.detach().cpu().numpy()
        assert a is None or isinstance(a,np.ndarray)
        return a
    
    return convert(args[0]) if len(args)==1 else tuple(convert(a) for a in args)

def save_mtl(filename):
    png_file = filename.with_suffix(".png")

    with open(filename, 'w') as file:
        file.write(f"newmtl material\n map_Kd {png_file}")

def save_obj(
        v:torch.Tensor,
        v_idx:torch.Tensor,
        t:torch.Tensor,
        t_idx:torch.Tensor,
        filename:Path
        ):
    
    filename = Path(filename)

    mtlfile = filename.with_suffix('.mtl')
    save_mtl(mtlfile)

    bytes_io = io.BytesIO()
    # np.savetxt(bytes_io, np.array([0, 1]),  f'mtllib {mtlfile}')

    np.savetxt(bytes_io, v.detach().cpu().numpy(), 'v %.4f %.4f %.4f', header=f"mtllib {mtlfile}", comments="")
    if t is not None:
        t = torch.stack((t[:, 0], 1.0 - t[:, 1]), dim=1)
        np.savetxt(bytes_io, t.cpu().numpy(), 'vt %f %f') #1-based indexing
    
    if t_idx is None:
        np.savetxt(bytes_io, v_idx.cpu().numpy() + 1, 'f %d %d %d') #1-based indexing
    else:
        fidx = torch.stack((v_idx, t_idx), dim=1) + 1
        fidx = torch.concat((fidx[:, :, 0], fidx[:, :, 1], fidx[:, :, 2]), dim=1)
        np.savetxt(bytes_io, fidx.cpu().numpy() , 'f %d/%d %d/%d %d/%d', header='usemtl material', comments="") #1-based indexing

    obj_path = filename.with_suffix('.obj')
    with open(obj_path, 'w') as file:
        file.write(bytes_io.getvalue().decode('UTF-8'))

# obj loading
def load_obj(filename:Path, device='cuda'):
    filename = Path(filename)
    obj_path = filename.with_suffix('.obj')
    with open(obj_path) as file:
        obj_text = file.read()
    num = r"([0-9\.\-eE]+)"

    # vertices
    v = re.findall(f"(v {num} {num} {num})",obj_text)
    vn = re.findall(f"(vn {num} {num} {num})", obj_text)
    vt = re.findall(f"(vt {num} {num})", obj_text)

    has_tex = len(vt) != 0
    has_nrm = len(vn) != 0

    # vertex position normal texcoord values
    vertices = np.array(v)[:, 1:].astype(np.float32)
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)

    vertex_normals = None
    if has_nrm:
        vertex_normals = np.array(vn)[:, 1:].astype(np.float32)
        vertex_normals = torch.tensor(vertex_normals, dtype=torch.float32, device=device)
    
    vertex_texcoords = None
    if has_tex:
        vertex_texcoords = np.array(vt)[:, 1:].astype(np.float32)
        vertex_texcoords = torch.tensor(vertex_texcoords, dtype=torch.float32, device=device)
        vertex_texcoords = torch.stack((vertex_texcoords[:, 0], 1.0 - vertex_texcoords[:, 1]), dim=1)

    # cases of face indices
    all_faces = []
    f = re.findall(f"(f {num} {num} {num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.int32).reshape(-1,3,1)[...,:1])
    f = re.findall(f"(f {num}/{num} {num}/{num} {num}/{num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.int32).reshape(-1,3,2)[...,:2])
    f = re.findall(f"(f {num}/{num}/{num} {num}/{num}/{num} {num}/{num}/{num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.int32).reshape(-1,3,3)[...,:3])
    f = re.findall(f"(f {num}//{num} {num}//{num} {num}//{num})",obj_text)
    if f:
        all_faces.append(np.array(f)[:,1:].astype(np.int32).reshape(-1,3,2)[...,:1])

    all_faces = np.concatenate(all_faces,axis=0)
    all_faces -= 1 #1-based indexing

    # indicies
    vertex_indices = all_faces[:,:,0]
    vertex_indices = torch.tensor(vertex_indices, dtype=torch.int32, device=device)

    vertex_texcoords_indices = None
    if has_tex:
        vertex_texcoords_indices = all_faces[:,:,1]
        vertex_texcoords_indices = torch.tensor(vertex_texcoords_indices, dtype=torch.int32, device=device)

    vertex_normals_indices = None
    if has_nrm:
        vertex_normals_indices = all_faces[:,:,-1]
        vertex_normals_indices = torch.tensor(vertex_normals_indices, dtype=torch.int32, device=device)

    return vertices, vertex_indices, vertex_normals, vertex_normals_indices, vertex_texcoords, vertex_texcoords_indices

def save_ply(
        filename:Path,
        vertices:torch.Tensor, #V,3
        faces:torch.Tensor, #F,3
        vertex_colors:torch.Tensor=None, #V,3
        vertex_normals:torch.Tensor=None, #V,3
        ):
        
    filename = Path(filename).with_suffix('.ply')
    vertices,faces,vertex_colors = to_numpy(vertices,faces,vertex_colors)
    assert np.all(np.isfinite(vertices)) and faces.min()==0 and faces.max()==vertices.shape[0]-1

    header = 'ply\nformat ascii 1.0\n'

    header += 'element vertex ' + str(vertices.shape[0]) + '\n'
    header += 'property double x\n'
    header += 'property double y\n'
    header += 'property double z\n'

    if vertex_normals is not None:
        header += 'property double nx\n'
        header += 'property double ny\n'
        header += 'property double nz\n'

    if vertex_colors is not None:
        assert vertex_colors.shape[0] == vertices.shape[0]
        color = (vertex_colors*255).astype(np.uint8)
        header += 'property uchar red\n'
        header += 'property uchar green\n'
        header += 'property uchar blue\n'

    header += 'element face ' + str(faces.shape[0]) + '\n'
    header += 'property list int int vertex_indices\n'
    header += 'end_header\n'

    with open(filename, 'w') as file:
        file.write(header)

        for i in range(vertices.shape[0]):
            s = f"{vertices[i,0]} {vertices[i,1]} {vertices[i,2]}"    
            if vertex_normals is not None:
                s += f" {vertex_normals[i,0]} {vertex_normals[i,1]} {vertex_normals[i,2]}"
            if vertex_colors is not None:
                s += f" {color[i,0]:03d} {color[i,1]:03d} {color[i,2]:03d}"
            file.write(s+'\n')
        
        for i in range(faces.shape[0]):
            file.write(f"3 {faces[i,0]} {faces[i,1]} {faces[i,2]}\n")
    full_verts = vertices[faces] #F,3,3
    
def save_images(
        images:torch.Tensor, #B,H,W,CH
        dir:Path,
        ):
    dir = Path(dir)
    dir.mkdir(parents=True,exist_ok=True)
    for i in range(images.shape[0]):
        imageio.imwrite(dir/f'{i:02d}.png',(images.detach()[i,:,:,:3]*255).clamp(max=255).type(torch.uint8).cpu().numpy())

def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def make_star_cameras(az_count,pol_count,distance:float=10.,r=None,image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    A = az_count
    P = pol_count
    C = A * P

    phi = torch.arange(0,A) * (2*torch.pi/A)
    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = torch.arange(1,P+1) * (torch.pi/(P+1)) - torch.pi/2
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv = _translation(0, 0, -distance, device) @ mv

    return mv, _projection(r,device)

import trimesh
def make_sphere(level:int=2,radius=1.,device='cuda') -> tuple[torch.Tensor,torch.Tensor]:
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.int32)
    return vertices,faces