# BSD 3-Clause License

# Copyright (c) 2023, CGLab, GIST

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn.functional as tfunc
import numpy as np
from torch_scatter import scatter, scatter_max, scatter_min, scatter_mean
import time

# 
# 
# from adaptive laplacian for inverse rendering
# 
# 

class Geometry:

    def __init__(self, v, f):
        '''
        for the caching
        '''
        self._v = v
        self._f = f
        self._nv = v.shape[0]
        self._nf = f.shape[0]
        self.device = v.device
        self._v_deg = None

        self._vertex_dualareas = None
        self._vertex_bandwidth_cotan = None
        self._vertex_bandwidth_uniform = None

        self._face_areas = None

        self._e = None
        self._ne = None
        self._edge_lengths = None

        self._indices = None
        self._coalesce_indices = None
        self._lap_uniform_values = None
        self._lap_cotangent_values = None
        self._lap_kernel_fix = None
        self._lap_kernel_ada = None
        self._lap_kernel_mix = None

        self._laplacian_uniform = None
        self._laplacian_cotangent = None
        self._laplacian_kernel = None
        self._laplacian_ada = None
        self._laplacian_mix = None
        self._laplacian_mixture = None

        self._adj = None
        self._mesh_length = None
        self._shape_scale = None

        self.local_min = None
        self.local_max = None
        self.local_scale = None

    def __calc_adjacency_list__(self):
        if self._adj is not None:
            return

        ii = self._f[:, [1, 2, 0]].flatten()
        jj = self._f[:, [2, 0, 1]].flatten()
        self._adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)

    def __calc_vertex_degree__(self):
        if self._v_deg is not None:
            return 
        self.__calc_adjacency_list__()

        self._v_deg = torch.unique(self._adj[0], return_counts=True)[1]

    def __calc_edges__(self):
        """
        from Continous remeshing. Palfinger 2022 CGI
        returns tuple of
        - edges E,2 long, 0 for unused, lower vertex index first
        - face_to_edge F,3 long
        - (optional) edge_to_face shape=E,[left,right],[face,side]

        o-<-----e1     e0,e1...edge, e0<e1
        |      /A      L,R....left and right face
        |  L /  |      both triangles ordered counter clockwise
        |  / R  |      normals pointing out of screen
        V/      |      
        e0---->-o     
        """
        if self._e is not None:
            return

        F = self._nf
        
        # make full edges, lower vertex index first
        face_edges = torch.stack((self._f, self._f.roll(-1,1)),dim=-1) #F*3,3,2
        full_edges = face_edges.reshape(F*3,2)
        sorted_edges,_ = full_edges.sort(dim=-1) #F*3,2 TODO min/max faster?

        # make unique edges
        self._e, full_to_unique = torch.unique(input=sorted_edges, sorted=True, return_inverse=True, dim=0) #(E,2),(F*3)
        self._ne = self._e.shape[0]
        self._face_to_edge = full_to_unique.reshape(F,3) #F,3

        is_right = full_edges[:,0]!=sorted_edges[:,0] #F*3
        edge_to_face = torch.zeros((self._ne,2,2), dtype=torch.long, device=self.device) #E,LR=2,S=2
        scatter_src = torch.cartesian_prod(torch.arange(0,F,device=self.device),torch.arange(0,3,device=self.device)) #F*3,2
        edge_to_face.reshape(2*self._ne,2).scatter_(dim=0,index=(2*full_to_unique+is_right)[:,None].expand(F*3,2),src=scatter_src) #E,LR=2,S=2
        edge_to_face[0] = 0
        self._edge_to_face = edge_to_face

    def __calc_edge_lengths__(self):
        if self._edge_lengths is not None:
            return 

        self.__calc_edges__()
        
        full_vertices = self._v[self._e] #E,2,3
        a,b = full_vertices.unbind(dim=1) #E,3
        self._edge_lengths = torch.norm(a-b,p=2,dim=-1)

    def __calc_mesh_length__(self):
        if self._mesh_length is not None:
            return 
        
        self.__calc_edge_lengths__()
        self._mesh_length = torch.sum(self._edge_lengths)/self._ne

    def __calc_faces_areas__(self):
        if self._face_areas is not None:
            return
        face_verts = self._v[self._f]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)

        s = 0.5 * (A + B + C)
        self._face_areas = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
    
    def __calc_vertex_dualareas__(self):
        if self._vertex_dualareas is not None:
            return
        self.__calc_faces_areas__()
        vertex_dualareas = torch.zeros((self._nv, 3),dtype=self._v.dtype,device=self.device) #V,C=3,3
        vertex_dualareas.scatter_add_(dim=0,index=self._f,src=self._face_areas[:, None].expand(self._nf, 3))
        self._vertex_dualareas = vertex_dualareas.sum(dim=1) / 3.0 #V,3

    def __calc_shape_scale__(self):
        if self._shape_scale is not None:
            return
        self.__calc_faces_areas__()
        self._shape_scale = torch.sqrt(torch.sum(self._face_areas))

    def __calc_sparse_coo_indices__(self):
        if self._indices is not None:
            return
        self.__calc_adjacency_list__()
        diag_idx = self._adj[0]
        self._indices = torch.cat((self._adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    
    def __calc_laplacian_uniform_indices_and_values(self, normalize=True, return_matrix=True):
        if self._lap_uniform_values is not None:
            return

        self.__calc_adjacency_list__()
        self.__calc_vertex_degree__()
        self.__calc_sparse_coo_indices__()

        adj = self._adj
        adj_values = torch.ones(adj.shape[1], dtype=torch.float, device=self.device)

        # normalization
        if normalize:        
            deg = self._v_deg
            adj_values = torch.div(adj_values[adj[1]], deg[adj[0]])

        # Diagonal indicess
        self._lap_uniform_values = torch.cat((-adj_values, adj_values))
        self._laplacian_uniform = torch.sparse_coo_tensor(self._indices, self._lap_uniform_values, (self._nv,self._nv)).coalesce()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_uniform.indices()
    
    def __calc_laplacian_cotangent_indices_and_values__(self, normalize=True):
        """
        https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
        """
        if self._laplacian_cotangent is not None:
            return
        
        face_verts = self._v[self._f]
        v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
        A = (v1 - v2).norm(dim=1)
        B = (v0 - v2).norm(dim=1)
        C = (v0 - v1).norm(dim=1)
        
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2)
        cotb = (A2 + C2 - B2)
        cotc = (A2 + B2 - C2)
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 2.0

        ii = self._f[:, [1, 2, 0]]
        jj = self._f[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, self._nf * 3)
        L = torch.sparse.FloatTensor(idx, cot.view(-1), (self._nv, self._nv))
        L += L.t()
        L = L.coalesce()

        values = torch.sparse.sum(L, dim=0).to_dense()
        sym_indices = torch.arange(self._nv, device=self.device)
        sym_indices = torch.stack([sym_indices, sym_indices], dim=0)

        if normalize:
            values = 1./values
            L_indices = L.indices()
            D = torch.sparse.FloatTensor(L_indices, L.values()*values[L_indices[0]], (self._nv, self._nv))
            L = torch.sparse.FloatTensor(sym_indices, torch.ones_like(values), (self._nv, self._nv)) - D
            self._laplacian_cotangent = L.coalesce()
            self._lap_cotangent_values = self._laplacian_cotangent.values()
            if self._coalesce_indices is None:
                self._coalesce_indices = self._laplacian_cotangent.indices()
        else:
            L = torch.sparse.FloatTensor(sym_indices, values, (self._nv, self._nv)) - L
            self._laplacian_cotangent = L.coalesce()
            self._lap_cotangent_values = self._laplacian_cotangent.values()
            if self._coalesce_indices is None:
                self._coalesce_indices = self._laplacian_cotangent.indices()
    
    def __calc_vertex_normalize_local__(self):

        if self.local_min is not None:
            return

        adj_verts = self._v[self._coalesce_indices[1]]

        mins, _ = scatter_min(src=adj_verts, index=self._coalesce_indices[0][:, None].expand(-1, 3), dim=0)
        maxs, _ = scatter_max(src=adj_verts, index=self._coalesce_indices[0][:, None].expand(-1, 3), dim=0)

        range_ = (maxs-mins)/2.0
        mins = mins-range_
        maxs = maxs+range_

        self.local_min, self.local_max = mins, maxs
    
    def __calc_laplacian_kernelized_indices_and_values__(self, bandwidth=None, normalize=True):
        
        if self._laplacian_kernel is not None:
            return

        self.__calc_sparse_coo_indices__()
        self.__calc_vertex_normalize_local__()
        
        h = bandwidth
        if h is None:
            self.__calc_vertex_degree__()
            self.__calc_edge_lengths__()
            h = scatter(self._edge_lengths, index=self._e[:, 1], dim=0)/self._v_deg
            h = torch.max(((h[self._adj[1]] + h[self._adj[0]])/2.0)**(0.4), torch.tensor([1e-8], device=self.device))

        mmrange = 1.0
        if normalize:
            mmrange = 2.0/(self.local_max[self._adj[0]] - self.local_min[self._adj[0]])
        local_v = (self._v[self._adj[0]]-self._v[self._adj[1]])*mmrange
        distance = (local_v).square().sum(dim=1)
        values = self.__gaussian(h, distance)/h

        summ = scatter(values, index=self._adj[1], dim=0)
        values = torch.div(values, summ[self._adj[1]])

        values = torch.cat((-values, values))
        L = torch.sparse.FloatTensor(self._indices, values, (self._nv, self._nv))

        self._laplacian_kernel = L.coalesce()
        self._lap_kernel_fix = self._laplacian_kernel.values()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_kernel.indices()
        return

    def __calc_asymtotic_bandwidth__(self, coeff, scale):
        self.__calc_vertex_dualareas__()
        self.__calc_mesh_length__()
        self.__calc_laplacian_cotangent_indices_and_values__(True)
        self.__calc_laplacian_kernelized_indices_and_values__(None, True)

        # local normalized coordinates
        g_scale = torch.norm(self.local_max-self.local_min, dim=1, p=2) * scale
        h_common = 3 * coeff * self._vertex_dualareas / (torch.pi)
        mmrange = 2.0/(self.local_max[self._coalesce_indices[0]] - self.local_min[self._coalesce_indices[0]])
        local_v = (self._v[self._coalesce_indices[0]]-self._v[self._coalesce_indices[1]])*mmrange

        # gather edge weights
        Wker = self._lap_kernel_fix
        Wcot = self._lap_cotangent_values

        Lu = scatter(src=Wker[:, None].expand(-1, 3)*local_v, index=self._coalesce_indices[0], dim=0, reduce='sum')
        Ln = scatter(src=Wcot[:, None].expand(-1, 3)*local_v, index=self._coalesce_indices[0], dim=0, reduce='sum') 

        Lu = torch.sum(torch.square(Lu), dim=1)
        hu = h_common/torch.max(torch.tensor([1e-8], device=self.device), Lu)
        hu = torch.pow(hu, 1.0/7.0)

        Ln = torch.sum(torch.square(Ln), dim=1)
        hn = h_common/torch.max(torch.tensor([1e-8], device=self.device), Ln)
        hn = torch.pow(hn, 1.0/7.0) 

        self._vertex_bandwidth_uniform = hu * g_scale
        self._vertex_bandwidth_cotan = hn * g_scale

    def __gaussian(
        self,
        h:torch.tensor,   # input bandwidth
        dist:torch.tensor # L2 distance between two vertices
        )->torch.tensor:

        const = 1.0/(4*torch.pi*h);
        return torch.exp(-1.0 * dist/(4.0*h)) * const
    
    def __calc_laplacian_adaptive_indices_and_values__(self, coeff, scale_range, cotan):
        '''
        None mixed version
        '''
        if self._laplacian_ada is not None:
            return

        self.__calc_asymtotic_bandwidth__(coeff, scale_range)
        self.__calc_sparse_coo_indices__()
        
        if cotan == True:
            h = self._vertex_bandwidth_cotan
        else:
            h = self._vertex_bandwidth_uniform

        distance = (self._v[self._adj[0]] - self._v[self._adj[1]]).square().sum(dim=1)
        w0 = self.__gaussian(h[self._adj[0]], distance)
        w1 = self.__gaussian(h[self._adj[1]], distance)

        val0 = w0/h[self._adj[0]]*self._vertex_dualareas[self._adj[0]]
        val1 = w1/h[self._adj[1]]*self._vertex_dualareas[self._adj[1]]
        adj_values = (val0*val1).sqrt()

        values = adj_values
        values = torch.cat((-values, values))
        L = torch.sparse.FloatTensor(self._indices, values, (self._nv, self._nv))

        self._laplacian_ada = L.coalesce()
        self._lap_kernel_ada = self._laplacian_ada.values()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_ada.indices()

    def __calc_laplacian_mixed_indices_and_values__(self, coeff, scale):
        '''
        weight mix
        '''
        if self._laplacian_mix is not None:
            return

        self.__calc_asymtotic_bandwidth__(coeff, scale)
        self.__calc_sparse_coo_indices__()

        h0 = self._vertex_bandwidth_cotan
        h1 = self._vertex_bandwidth_uniform

        distance = (self._v[self._adj[0]] - self._v[self._adj[1]]).square().sum(dim=1)
        w00 = self.__gaussian(h0[self._adj[0]], distance)
        w01 = self.__gaussian(h0[self._adj[1]], distance)
        w10 = self.__gaussian(h1[self._adj[0]], distance)
        w11 = self.__gaussian(h1[self._adj[1]], distance)

        w002, w012 = w00, w01
        w102, w112 = w10, w11
        Wsum0, Wsum1 = w002+w102, w012+w112 
 
        val1 = (w00*w002/Wsum0/h0[self._adj[0]] + w10*w102/Wsum0/h1[self._adj[0]])*self._vertex_dualareas[self._adj[0]]
        val2 = (w01*w012/Wsum1/h0[self._adj[1]] + w11*w112/Wsum1/h1[self._adj[1]])*self._vertex_dualareas[self._adj[1]]
        adj_values = torch.sqrt(val1*val2)

        values = adj_values
        values = torch.cat((-values, values))
        L = torch.sparse.FloatTensor(self._indices, values, (self._nv, self._nv))

        self._laplacian_mix = L.coalesce()
        self._lap_kernel_mix = self._laplacian_mix.values()
        if self._coalesce_indices is None:
            self._coalesce_indices = self._laplacian_mix.indices()

    def __calc_sparse_coo__(self, indices, values):
        return torch.sparse_coo_tensor(indices, values, (self._nv, self._nv)).coalesce()

    '''    
    public functions for real usage
    '''
    def laplacian_uniform(self):
        if self._laplacian_uniform is not None:
            return self._laplacian_uniform

        self.__calc_laplacian_uniform_indices_and_values()
        return self._laplacian_uniform
    
    def laplacian_cotangent(self):
        if self._laplacian_cotangent is not None:
            return self._laplacian_cotangent

        self.__calc_laplacian_cotangent_indices_and_values__()
        return self._laplacian_cotangent
    
    def laplacian_adaptive(self, weight, scale):
        if self._laplacian_mix is not None:
            return self._laplacian_mix.coalesce()
        self.__calc_laplacian_mixed_indices_and_values__(weight, scale)
        return self._laplacian_mix.coalesce()

# accessing functions: use here
def vertex_dualareas(verts, faces):
    geom = Geometry(verts, faces)
    geom.__calc_vertex_dualareas__()
    return geom._vertex_dualareas

def asymtotic_bandwidth(verts, faces, smoothing_weight):
    geom = Geometry(verts, faces)
    geom.__calc_asymtotic_bandwidth__(smoothing_weight)
    return geom._vertex_bandwidth_cotan, geom._vertex_bandwidth_uniform

def laplacian_uniform(verts, faces):
    geom = Geometry(verts, faces)
    return geom.laplacian_uniform()

def laplacian_cotangent(verts, faces, weight=None):
    geom = Geometry(verts, faces)
    return geom.laplacian_cotangent()

def laplacian_adaptive(verts, faces, weight, scale):
    geom = Geometry(verts, faces)
    return geom.laplacian_adaptive(weight, scale)

def csc_cpu_to_coo_gpu(csc_cpu):
    coo = csc_cpu.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    ind = torch.tensor(indices, dtype=torch.long)
    val = torch.tensor(values, dtype=torch.float64)
    L = torch.sparse_coo_tensor(ind, val, coo.shape).cuda()
    return L.coalesce()

# 
# 
# from continuous remeshing, 2021, palfinger
# 
# 
def normalize_vertices(
        vertices:torch.Tensor, #V,3
    ):
    """shift and resize mesh to fit into a unit sphere"""
    vertices -= (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
    vertices /= torch.norm(vertices, dim=-1).max()
    return vertices


def prepend_dummies(
        vertices:torch.Tensor, #V,D
        faces:torch.Tensor, #F,3 long
    )->tuple[torch.Tensor,torch.Tensor]:
    """prepend dummy elements to vertices and faces to enable "masked" scatter operations"""
    V,D = vertices.shape
    vertices = torch.concat((torch.full((1,D),fill_value=torch.nan,device=vertices.device),vertices),dim=0)
    faces = torch.concat((torch.zeros((1,3),dtype=torch.long,device=faces.device),faces+1),dim=0)
    return vertices,faces

def remove_dummies(
        vertices:torch.Tensor, #V,D - first vertex all nan and unreferenced
        faces:torch.Tensor, #F,3 long - first face all zeros
    )->tuple[torch.Tensor,torch.Tensor]:
    """remove dummy elements added with prepend_dummies()"""
    return vertices[1:],faces[1:]-1

def calc_edges(
        faces: torch.Tensor,  # F,3 long - first face may be dummy with all zeros
        with_edge_to_face: bool = False
    ) -> tuple[torch.Tensor, ...]:
    """
    returns tuple of
    - edges E,2 long, 0 for unused, lower vertex index first
    - face_to_edge F,3 long
    - (optional) edge_to_face shape=E,[left,right],[face,side]

    o-<-----e1     e0,e1...edge, e0<e1
    |      /A      L,R....left and right face
    |  L /  |      both triangles ordered counter clockwise
    |  / R  |      normals pointing out of screen
    V/      |      
    e0---->-o     
    """

    F = faces.shape[0]
    
    # make full edges, lower vertex index first
    face_edges = torch.stack((faces,faces.roll(-1,1)),dim=-1) #F*3,3,2
    full_edges = face_edges.reshape(F*3,2)
    sorted_edges,_ = full_edges.sort(dim=-1) #F*3,2 TODO min/max faster?

    # make unique edges
    edges,full_to_unique = torch.unique(input=sorted_edges,sorted=True,return_inverse=True,dim=0) #(E,2),(F*3)
    E = edges.shape[0]
    face_to_edge = full_to_unique.reshape(F,3) #F,3

    if not with_edge_to_face:
        return edges, face_to_edge

    is_right = full_edges[:,0]!=sorted_edges[:,0] #F*3
    edge_to_face = torch.zeros((E,2,2),dtype=torch.long,device=faces.device) #E,LR=2,S=2
    scatter_src = torch.cartesian_prod(torch.arange(0,F,device=faces.device),torch.arange(0,3,device=faces.device)) #F*3,2
    edge_to_face.reshape(2*E,2).scatter_(dim=0,index=(2*full_to_unique+is_right)[:,None].expand(F*3,2),src=scatter_src) #E,LR=2,S=2
    edge_to_face[0] = 0
    return edges, face_to_edge, edge_to_face

def calc_edge_length(
        vertices:torch.Tensor, #V,3 first may be dummy
        edges:torch.Tensor, #E,2 long, lower vertex index first, (0,0) for unused
        )->torch.Tensor: #E

    full_vertices = vertices[edges] #E,2,3
    a,b = full_vertices.unbind(dim=1) #E,3
    return torch.norm(a-b,p=2,dim=-1)

def calc_face_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    """
    full_vertices = vertices[faces] #F,C=3,3
    v0,v1,v2 = full_vertices.unbind(dim=1) #F,3
    face_normals = torch.cross(v1-v0,v2-v0, dim=1) #F,3
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
    return face_normals #F,3

def calc_vertex_normals(
        vertices:torch.Tensor, #V,3 first vertex may be unreferenced
        faces:torch.Tensor, #F,3 long, first face may be all zero
        face_normals:torch.Tensor=None, #F,3, not normalized
        )->torch.Tensor: #F,3

    F = faces.shape[0]

    if face_normals is None:
        face_normals = calc_face_normals(vertices,faces)
    
    vertex_normals = torch.zeros((vertices.shape[0],3,3),dtype=vertices.dtype,device=vertices.device) #V,C=3,3
    vertex_normals.scatter_add_(dim=0,index=faces[:,:,None].expand(F,3,3),src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1) #V,3
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)

def calc_face_ref_normals(
        faces:torch.Tensor, #F,3 long, 0 for unused
        vertex_normals:torch.Tensor, #V,3 first unused
        normalize:bool=False,
        )->torch.Tensor: #F,3
    """calculate reference normals for face flip detection"""
    full_normals = vertex_normals[faces] #F,C=3,3
    ref_normals = full_normals.sum(dim=1) #F,3
    if normalize:
        ref_normals = tfunc.normalize(ref_normals, eps=1e-6, dim=1)
    return ref_normals

def pack(
        vertices:torch.Tensor, #V,3 first unused and nan
        faces:torch.Tensor, #F,3 long, 0 for unused
        )->tuple[torch.Tensor,torch.Tensor]: #(vertices,faces), keeps first vertex unused
    """removes unused elements in vertices and faces"""
    V = vertices.shape[0]
    
    # remove unused faces
    used_faces = faces[:,0]!=0
    used_faces[0] = True
    faces = faces[used_faces] #sync

    # remove unused vertices
    used_vertices = torch.zeros(V,3,dtype=torch.bool,device=vertices.device)
    used_vertices.scatter_(dim=0,index=faces,value=True,reduce='add') #TODO int faster?
    used_vertices = used_vertices.any(dim=1)
    used_vertices[0] = True
    vertices = vertices[used_vertices] #sync

    # update used faces
    ind = torch.zeros(V,dtype=torch.long,device=vertices.device)
    V1 = used_vertices.sum()
    ind[used_vertices] =  torch.arange(0,V1,device=vertices.device) #sync
    faces = ind[faces]

    return vertices,faces

def split_edges(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long, 0 for unused
        edges:torch.Tensor, #E,2 long 0 for unused, lower vertex index first
        face_to_edge:torch.Tensor, #F,3 long 0 for unused
        splits, #E bool
        pack_faces:bool=True,
        )->tuple[torch.Tensor,torch.Tensor]: #(vertices,faces)

    #   c2                    c2               c...corners = faces
    #    . .                   . .             s...side_vert, 0 means no split
    #    .   .                 .N2 .           S...shrunk_face
    #    .     .               .     .         Ni...new_faces
    #   s2      s1           s2|c2...s1|c1
    #    .        .            .     .  .
    #    .          .          . S .      .
    #    .            .        . .     N1    .
    #   c0...(s0=0)....c1    s0|c0...........c1
    #
    # pseudo-code:
    #   S = [s0|c0,s1|c1,s2|c2] example:[c0,s1,s2]
    #   split = side_vert!=0 example:[False,True,True]
    #   N0 = split[0]*[c0,s0,s2|c2] example:[0,0,0]
    #   N1 = split[1]*[c1,s1,s0|c0] example:[c1,s1,c0]
    #   N2 = split[2]*[c2,s2,s1|c1] example:[c2,s2,s1]

    V = vertices.shape[0]
    F = faces.shape[0]
    S = splits.sum().item() #sync

    if S==0:
        return vertices,faces
    
    edge_vert = torch.zeros_like(splits, dtype=torch.long) #E
    edge_vert[splits] = torch.arange(V,V+S,dtype=torch.long,device=vertices.device) #E 0 for no split, sync
    side_vert = edge_vert[face_to_edge] #F,3 long, 0 for no split
    split_edges = edges[splits] #S sync

    #vertices
    split_vertices = vertices[split_edges].mean(dim=1) #S,3
    vertices = torch.concat((vertices,split_vertices),dim=0)

    #faces
    side_split = side_vert!=0 #F,3
    shrunk_faces = torch.where(side_split,side_vert,faces) #F,3 long, 0 for no split
    new_faces = side_split[:,:,None] * torch.stack((faces,side_vert,shrunk_faces.roll(1,dims=-1)),dim=-1) #F,N=3,C=3
    faces = torch.concat((shrunk_faces,new_faces.reshape(F*3,3))) #4F,3
    if pack_faces:
        mask = faces[:,0]!=0
        mask[0] = True
        faces = faces[mask] #F',3 sync

    return vertices,faces

def collapse_edges(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long 0 for unused
        edges:torch.Tensor, #E,2 long 0 for unused, lower vertex index first
        priorities:torch.Tensor, #E float
        stable:bool=False, #only for unit testing
        )->tuple[torch.Tensor,torch.Tensor]: #(vertices,faces)
        
    V = vertices.shape[0]
    
    # check spacing
    _,order = priorities.sort(stable=stable) #E
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0,len(rank),device=rank.device)
    vert_rank = torch.zeros(V,dtype=torch.long,device=vertices.device) #V
    edge_rank = rank #E
    for i in range(3):
        scatter_max(src=edge_rank[:,None].expand(-1,2).reshape(-1),index=edges.reshape(-1),dim=0,out=vert_rank)
        edge_rank,_ = vert_rank[edges].max(dim=-1) #E
    candidates = edges[(edge_rank==rank).logical_and_(priorities>0)] #E',2

    # check connectivity
    vert_connections = torch.zeros(V,dtype=torch.long,device=vertices.device) #V
    vert_connections[candidates[:,0]] = 1 #start
    edge_connections = vert_connections[edges].sum(dim=-1) #E, edge connected to start
    vert_connections.scatter_add_(dim=0,index=edges.reshape(-1),src=edge_connections[:,None].expand(-1,2).reshape(-1))# one edge from start
    vert_connections[candidates] = 0 #clear start and end
    edge_connections = vert_connections[edges].sum(dim=-1) #E, one or two edges from start
    vert_connections.scatter_add_(dim=0,index=edges.reshape(-1),src=edge_connections[:,None].expand(-1,2).reshape(-1)) #one or two edges from start
    collapses = candidates[vert_connections[candidates[:,1]] <= 2] # E" not more than two connections between start and end

    # mean vertices
    vertices[collapses[:,0]] = vertices[collapses].mean(dim=1) #TODO dim?

    # update faces
    dest = torch.arange(0,V,dtype=torch.long,device=vertices.device) #V
    dest[collapses[:,1]] = dest[collapses[:,0]]
    faces = dest[faces] #F,3 TODO optimize?
    c0,c1,c2 = faces.unbind(dim=-1)
    collapsed = (c0==c1).logical_or_(c1==c2).logical_or_(c0==c2)
    faces[collapsed] = 0

    return vertices,faces

def calc_face_collapses(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long, 0 for unused
        edges:torch.Tensor, #E,2 long 0 for unused, lower vertex index first
        face_to_edge:torch.Tensor, #F,3 long 0 for unused
        edge_length:torch.Tensor, #E
        face_normals:torch.Tensor, #F,3
        vertex_normals:torch.Tensor, #V,3 first unused
        min_edge_length:torch.Tensor=None, #V
        area_ratio = 0.5, #collapse if area < min_edge_length**2 * area_ratio
        shortest_probability = 0.8
        )->torch.Tensor: #E edges to collapse
    
    E = edges.shape[0]
    F = faces.shape[0]

    # face flips
    ref_normals = calc_face_ref_normals(faces,vertex_normals,normalize=False) #F,3
    face_collapses = (face_normals*ref_normals).sum(dim=-1)<0 #F
    
    # small faces
    if min_edge_length is not None:
        min_face_length = min_edge_length[faces].mean(dim=-1) #F
        min_area = min_face_length**2 * area_ratio #F
        face_collapses.logical_or_(face_normals.norm(dim=-1) < min_area*2) #F
        face_collapses[0] = False

    # faces to edges
    face_length = edge_length[face_to_edge] #F,3

    if shortest_probability<1:
        #select shortest edge with shortest_probability chance
        randlim = round(2/(1-shortest_probability))
        rand_ind = torch.randint(0,randlim,size=(F,),device=faces.device).clamp_max_(2) #selected edge local index in face
        sort_ind = torch.argsort(face_length,dim=-1,descending=True) #F,3
        local_ind = sort_ind.gather(dim=-1,index=rand_ind[:,None])
    else:
        local_ind = torch.argmin(face_length,dim=-1)[:,None] #F,1 0...2 shortest edge local index in face
    
    edge_ind = face_to_edge.gather(dim=1,index=local_ind)[:,0] #F 0...E selected edge global index
    edge_collapses = torch.zeros(E,dtype=torch.long,device=vertices.device)
    edge_collapses.scatter_add_(dim=0,index=edge_ind,src=face_collapses.long()) #TODO legal for bool?

    return edge_collapses.bool()

def flip_edges(
        vertices:torch.Tensor, #V,3 first unused
        faces:torch.Tensor, #F,3 long, first must be 0, 0 for unused
        edges:torch.Tensor, #E,2 long, first must be 0, 0 for unused, lower vertex index first
        edge_to_face:torch.Tensor, #E,[left,right],[face,side]
        with_border:bool=True, #handle border edges (D=4 instead of D=6)
        with_normal_check:bool=True, #check face normal flips
        stable:bool=False, #only for unit testing
        ):
    V = vertices.shape[0]
    E = edges.shape[0]
    device=vertices.device
    vertex_degree = torch.zeros(V,dtype=torch.long,device=device) #V long
    vertex_degree.scatter_(dim=0,index=edges.reshape(E*2),value=1,reduce='add')
    neighbor_corner = (edge_to_face[:,:,1] + 2) % 3 #go from side to corner
    neighbors = faces[edge_to_face[:,:,0],neighbor_corner] #E,LR=2
    edge_is_inside = neighbors.all(dim=-1) #E

    if with_border:
        # inside vertices should have D=6, border edges D=4, so we subtract 2 for all inside vertices
        # need to use float for masks in order to use scatter(reduce='multiply')
        vertex_is_inside = torch.ones(V,2,dtype=torch.float32,device=vertices.device) #V,2 float
        src = edge_is_inside.type(torch.float32)[:,None].expand(E,2) #E,2 float
        vertex_is_inside.scatter_(dim=0,index=edges,src=src,reduce='multiply')
        vertex_is_inside = vertex_is_inside.prod(dim=-1,dtype=torch.long) #V long
        vertex_degree -= 2 * vertex_is_inside #V long

    neighbor_degrees = vertex_degree[neighbors] #E,LR=2
    edge_degrees = vertex_degree[edges] #E,2
    #
    # loss = Sum_over_affected_vertices((new_degree-6)**2)
    # loss_change = Sum_over_neighbor_vertices((degree+1-6)**2-(degree-6)**2)
    #                   + Sum_over_edge_vertices((degree-1-6)**2-(degree-6)**2)
    #             = 2 * (2 + Sum_over_neighbor_vertices(degree) - Sum_over_edge_vertices(degree))
    #
    loss_change = 2 + neighbor_degrees.sum(dim=-1) - edge_degrees.sum(dim=-1) #E
    candidates = torch.logical_and(loss_change<0, edge_is_inside) #E
    loss_change = loss_change[candidates] #E'
    if loss_change.shape[0]==0:
        return

    edges_neighbors = torch.concat((edges[candidates],neighbors[candidates]),dim=-1) #E',4
    _,order = loss_change.sort(descending=True, stable=stable) #E'
    rank = torch.zeros_like(order)
    rank[order] = torch.arange(0,len(rank),device=rank.device)
    vertex_rank = torch.zeros((V,4),dtype=torch.long,device=device) #V,4
    scatter_max(src=rank[:,None].expand(-1,4),index=edges_neighbors,dim=0,out=vertex_rank)
    vertex_rank,_ = vertex_rank.max(dim=-1) #V
    neighborhood_rank,_ = vertex_rank[edges_neighbors].max(dim=-1) #E'
    flip = rank==neighborhood_rank #E'

    if with_normal_check:
        #  cl-<-----e1     e0,e1...edge, e0<e1
        #   |      /A      L,R....left and right face
        #   |  L /  |      both triangles ordered counter clockwise
        #   |  / R  |      normals pointing out of screen
        #   V/      |      
        #   e0---->-cr    
        v = vertices[edges_neighbors] #E",4,3
        v = v - v[:,0:1] #make relative to e0 
        e1 = v[:,1]
        cl = v[:,2]
        cr = v[:,3]
        n = torch.cross(e1,cl) + torch.cross(cr,e1) #sum of old normal vectors 
        flip.logical_and_(torch.sum(n*torch.cross(cr,cl),dim=-1)>0) #first new face
        flip.logical_and_(torch.sum(n*torch.cross(cl-e1,cr-e1),dim=-1)>0) #second new face

    flip_edges_neighbors = edges_neighbors[flip] #E",4
    flip_edge_to_face = edge_to_face[candidates,:,0][flip] #E",2
    flip_faces = flip_edges_neighbors[:,[[0,3,2],[1,2,3]]] #E",2,3
    faces.scatter_(dim=0,index=flip_edge_to_face.reshape(-1,1).expand(-1,3),src=flip_faces.reshape(-1,3))

def remesh(
        vertices_etc:torch.Tensor, #V,D
        faces:torch.Tensor, #F,3 long
        min_edgelen:torch.Tensor, #V
        max_edgelen:torch.Tensor, #V
        flip:bool,
        max_vertices=1e6
        ):

    # dummies
    vertices_etc,faces = prepend_dummies(vertices_etc,faces)
    vertices = vertices_etc[:,:3] #V,3
    nan_tensor = torch.tensor([torch.nan],device=min_edgelen.device)
    min_edgelen = torch.concat((nan_tensor,min_edgelen))
    max_edgelen = torch.concat((nan_tensor,max_edgelen))

    # collapse
    edges,face_to_edge = calc_edges(faces) #E,2 F,3
    edge_length = calc_edge_length(vertices,edges) #E
    face_normals = calc_face_normals(vertices,faces,normalize=False) #F,3
    vertex_normals = calc_vertex_normals(vertices,faces,face_normals) #V,3
    face_collapse = calc_face_collapses(vertices,faces,edges,face_to_edge,edge_length,face_normals,vertex_normals,min_edgelen,area_ratio=0.5)
    shortness = (1 - edge_length / min_edgelen[edges].mean(dim=-1)).clamp_min_(0) #e[0,1] 0...ok, 1...edgelen=0
    priority = face_collapse.float() + shortness
    vertices_etc,faces = collapse_edges(vertices_etc,faces,edges,priority)

    # split
    if vertices.shape[0]<max_vertices:
        edges,face_to_edge = calc_edges(faces) #E,2 F,3
        vertices = vertices_etc[:,:3] #V,3
        edge_length = calc_edge_length(vertices,edges) #E
        splits = edge_length > max_edgelen[edges].mean(dim=-1)
        vertices_etc,faces = split_edges(vertices_etc,faces,edges,face_to_edge,splits,pack_faces=False)

    vertices_etc,faces = pack(vertices_etc,faces)
    vertices = vertices_etc[:,:3]

    if flip:
        edges,_,edge_to_face = calc_edges(faces,with_edge_to_face=True) #E,2 F,3
        flip_edges(vertices,faces,edges,edge_to_face,with_border=False)

    return remove_dummies(vertices_etc,faces)

def lerp_unbiased(a:torch.Tensor,b:torch.Tensor,weight:float,step:int):
    """lerp with adam's bias correction"""
    c_prev = 1-weight**(step-1)
    c = 1-weight**step
    a_weight = weight*c_prev/c
    b_weight = (1-weight)/c
    a.mul_(a_weight).add_(b, alpha=b_weight)


class MeshOptimizer:
    """Use this like a pytorch Optimizer, but after calling opt.step(), do vertices,faces = opt.remesh()."""

    def __init__(self, 
            vertices:torch.Tensor, #V,3
            faces:torch.Tensor, #F,3
            lr=0.3, #learning rate
            betas=(0.8,0.8,0), #betas[0:2] are the same as in Adam, betas[2] may be used to time-smooth the relative velocity nu
            gammas=(0,0,0), #optional spatial smoothing for m1,m2,nu, values between 0 (no smoothing) and 1 (max. smoothing)
            nu_ref=0.3, #reference velocity for edge length controller
            edge_len_lims=(.01,.15), #smallest and largest allowed reference edge length
            edge_len_tol=.5, #edge length tolerance for split and collapse
            gain=.2,  #gain value for edge length controller
            laplacian_weight=.02, #for laplacian smoothing/regularization
            ramp=1, #learning rate ramp, actual ramp width is ramp/(1-betas[0])
            grad_lim=10., #gradients are clipped to m1.abs()*grad_lim
            remesh_interval=1, #larger intervals are faster but with worse mesh quality
            local_edgelen=True, #set to False to use a global scalar reference edge length instead
            max_vertices=1e6
            ):
        self._vertices = vertices
        self._faces = faces
        self._lr = lr
        self._betas = betas
        self._gammas = gammas
        self._nu_ref = nu_ref
        self._edge_len_lims = edge_len_lims
        self._edge_len_tol = edge_len_tol
        self._gain = gain
        self._laplacian_weight = laplacian_weight
        self._ramp = ramp
        self._grad_lim = grad_lim
        self._remesh_interval = remesh_interval
        self._local_edgelen = local_edgelen
        self._max_vertices = max_vertices
        self._step = 0
        self._start = time.time()

        V = self._vertices.shape[0]
        # prepare continuous tensor for all vertex-based data 
        self._vertices_etc = torch.zeros([V,9],device=vertices.device)
        self._split_vertices_etc()
        self.vertices.copy_(vertices) #initialize vertices
        self._vertices.requires_grad_()
        self._ref_len.fill_(edge_len_lims[1])

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    def _split_vertices_etc(self):
        self._vertices = self._vertices_etc[:,:3]
        self._m2 = self._vertices_etc[:,3]
        self._nu = self._vertices_etc[:,4]
        self._m1 = self._vertices_etc[:,5:8]
        self._ref_len = self._vertices_etc[:,8]
        
        with_gammas = any(g!=0 for g in self._gammas)
        self._smooth = self._vertices_etc[:,:8] if with_gammas else self._vertices_etc[:,:3]

    def zero_grad(self):
        self._vertices.grad = None

    @torch.no_grad()
    def step(self):
        
        eps = 1e-8

        self._step += 1

        # spatial smoothing
        edges,_ = calc_edges(self._faces) #E,2
        E = edges.shape[0]
        edge_smooth = self._smooth[edges] #E,2,S
        neighbor_smooth = torch.zeros_like(self._smooth) #V,S
        scatter_mean(src=edge_smooth.flip(dims=[1]).reshape(E*2,-1),index=edges.reshape(E*2,1),dim=0,out=neighbor_smooth)
        
        #apply optional smoothing of m1,m2,nu
        if self._gammas[0]:
            self._m1.lerp_(neighbor_smooth[:,5:8],self._gammas[0])
        if self._gammas[1]:
            self._m2.lerp_(neighbor_smooth[:,3],self._gammas[1])
        if self._gammas[2]:
            self._nu.lerp_(neighbor_smooth[:,4],self._gammas[2])

        #add laplace smoothing to gradients
        laplace = self._vertices - neighbor_smooth[:,:3]
        grad = torch.addcmul(self._vertices.grad, laplace, self._nu[:,None], value=self._laplacian_weight)

        #gradient clipping
        if self._step>1:
            grad_lim = self._m1.abs().mul_(self._grad_lim)
            grad.clamp_(min=-grad_lim,max=grad_lim)

        # moment updates
        lerp_unbiased(self._m1, grad, self._betas[0], self._step)
        lerp_unbiased(self._m2, (grad**2).sum(dim=-1), self._betas[1], self._step)

        velocity = self._m1 / self._m2[:,None].sqrt().add_(eps) #V,3
        speed = velocity.norm(dim=-1) #V

        if self._betas[2]:
            lerp_unbiased(self._nu,speed,self._betas[2],self._step) #V
        else:
            self._nu.copy_(speed) #V

        # update vertices
        ramped_lr = self._lr * min(1,self._step * (1-self._betas[0]) / self._ramp)
        self._vertices.add_(velocity * self._ref_len[:,None], alpha=-ramped_lr)

        # update target edge length
        if self._step % self._remesh_interval == 0:
            if self._local_edgelen:
                len_change = (1 + (self._nu - self._nu_ref) * self._gain)
            else:
                len_change = (1 + (self._nu.mean() - self._nu_ref) * self._gain)
            self._ref_len *= len_change
            self._ref_len.clamp_(*self._edge_len_lims)

    def remesh(self, flip:bool=True)->tuple[torch.Tensor,torch.Tensor]:
        min_edge_len = self._ref_len * (1 - self._edge_len_tol)
        max_edge_len = self._ref_len * (1 + self._edge_len_tol)
            
        self._vertices_etc,self._faces = remesh(self._vertices_etc,self._faces,min_edge_len,max_edge_len,flip,self._max_vertices)

        self._split_vertices_etc()
        self._vertices.requires_grad_()

        return self._vertices, self._faces

# 
# libigl support
# 
def unwrap(v, f):
    import igl
    b = np.array([2, 1])

    bnd = igl.boundary_loop(f)

    b[0] = bnd[0]
    b[1] = bnd[int(bnd.size / 2)]

    bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    # LSCM parametrization
    _, uv = igl.lscm(v, f.astype(dtype=np.int32), b, bc)

    return uv.astype(dtype=np.float32)

# testing
if __name__ == "__main__":
    pos = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    ind = [[0, 1, 2]]

    v, f = torch.tensor(pos, dtype=torch.float64), torch.tensor(ind, dtype=torch.long)
    
    cot = laplacian_cotangent(v, f)
    ada = laplacian_adaptive(v, f, 0.96, 0.1)