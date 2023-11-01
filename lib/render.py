import nvdiffrast.torch as dr
import torch
import torch.nn.functional as tfunc

def _warmup(glctx):
    #windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])

class Rasterizer:

    _glctx:dr.RasterizeGLContext = None
    
    def __init__(
            self,
            mv: torch.Tensor, #C,4,4
            proj: torch.Tensor, #C,4,4
            image_size: tuple[int,int],
            ):
        self._mvp = proj @ mv #C,4,4
        self._campos = torch.linalg.inv(mv)[:, :3, 3]
        self._image_size = image_size
        self._glctx = dr.RasterizeGLContext()
        _warmup(self._glctx)

    def tangent(self, v_pos, v_idx, v_nrm, n_idx, v_tex, t_idx):
        vertices = v_pos[v_idx.type(dtype=torch.long)] #F,C=3,3
        texcoords = v_tex[t_idx.type(dtype=torch.long)]
        
        v0, v1, v2 = vertices.unbind(dim=1) #F,3
        t0, t1, t2 = texcoords.unbind(dim=1)

        dp0, dp1 = v1 - v0, v2 - v0
        duv0, duv1 = t1 - t0, t2 - t0
        
        r = (duv0[..., 0:1] * duv1[..., 1:2] - duv0[..., 1:2] * duv1[..., 0:1])
        tangent = (dp0 * duv1[..., 1:2] - dp1 * duv0[..., 1:2])/r

        vertex_tangent = torch.zeros((v_pos.shape[0],3,3), dtype=v_pos.dtype, device=v_pos.device) #V,C=3,3
        vertex_tangent.scatter_add_(dim=0, index=n_idx.type(dtype=torch.int64)[:,:,None].expand(-1,3,3), src=tangent[:,None,:].expand(-1,3,3))
        vertex_tangent = vertex_tangent.sum(dim=1) #V,3

        vertex_tangent = tfunc.normalize(vertex_tangent)
        vertex_tangent = tfunc.normalize(vertex_tangent - torch.sum(vertex_tangent*v_nrm, -1, keepdim=True) * v_nrm)

        return vertex_tangent

    def render_normal(self,
            v_pos: torch.Tensor, #V,3 float
            faces: torch.Tensor, #F,3 long
            v_nrm: torch.Tensor, #V,3 float
            n_idx: torch.Tensor, 
            ) ->torch.Tensor: #C,H,W  ,4

        V = v_pos.shape[0]
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((v_pos, torch.ones(V,1,device=v_pos.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out,_ = dr.rasterize(self._glctx, vertices_clip, faces, resolution=self._image_size, grad_db=False) #C,H,W,4
        vert_col = v_nrm
        color,_ = dr.interpolate(vert_col, rast_out, n_idx) #C,H,W,3
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        color = torch.concat((color,alpha),dim=-1) #C,H,W,4
        color = dr.antialias(color, rast_out, vertices_clip, faces) #C,H,W,4
        return color #C,H,W,4

    def render_texture_albedo(self, 
            v_pos: torch.Tensor, #V,3 float
            v_idx: torch.Tensor, #F,3 long // faces
            v_tex: torch.Tensor,
            t_idx: torch.Tensor, 
            texture: torch.Tensor, 
            resolution = None,
            ) -> torch.Tensor: #C,H,W,4
        
        V = v_pos.shape[0]
        v_idx = v_idx.type(torch.int32)
        vert_hom = torch.cat((v_pos, torch.ones(V,1,device=v_pos.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        if resolution is None:
            resolution = self._image_size
        rast_out, rast_out_db = dr.rasterize(self._glctx, vertices_clip, v_idx, resolution=resolution, grad_db=False) #C,H,W,4
         
        texc, texd = dr.interpolate(v_tex[None, ...], rast_out, t_idx, rast_db=rast_out_db, diff_attrs='all')
        if isinstance(texture["albedo"], list):
            albedo = dr.texture(texture["albedo"][0], texc, filter_mode='linear')
        else:
            albedo = dr.texture(texture["albedo"][None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=6)
        color = albedo

        color = color * torch.clamp(rast_out[..., -1:], 0, 1) #C,H,W,1
        return color

    def render_texture_normal(self,
            v_pos: torch.Tensor, #V,3 float
            v_idx: torch.Tensor, #F,3 long // faces
            v_nrm: torch.Tensor, #V,3 float
            n_idx: torch.Tensor,
            v_tex: torch.Tensor,
            t_idx: torch.Tensor, 
            texture: torch.Tensor, 
            tangent_space = False,
            ) -> torch.Tensor: #C,H,W,4

        V = v_pos.shape[0]
        v_idx = v_idx.type(torch.int32)
        vert_hom = torch.cat((v_pos, torch.ones(V,1,device=v_pos.device)),axis=-1) #V,3 -> V,4
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) #C,V,4
        rast_out, rast_out_db = dr.rasterize(self._glctx, vertices_clip, v_idx, resolution=self._image_size, grad_db=False) #C,H,W,4
    
        # View pos
        rast_pos,_ = dr.interpolate(v_pos.contiguous(), rast_out, v_idx.int())
        # view_vec = tfunc.normalize(self._campos[:, None, None, :] - rast_pos)

        texc, texd = dr.interpolate(v_tex[None, ...], rast_out, t_idx, rast_db=rast_out_db, diff_attrs='all')
        if isinstance(texture["normal"], list):
            normal = dr.texture(texture["normal"][0], texc, filter_mode='linear')
        else:
            normal = dr.texture(texture["normal"][None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=6)

        # TBN
        v_nrm = (v_nrm + 1)/2 #V,3

        if tangent_space:
            T = self.tangent(v_pos, v_idx, v_nrm, n_idx, v_tex, t_idx)
            B = tfunc.normalize(torch.cross(T, v_nrm), dim=-1)
            T, tb = dr.interpolate(T, rast_out, v_idx, rast_db=rast_out_db, diff_attrs='all') #C,H,W,3
            B, Bb = dr.interpolate(B, rast_out, v_idx, rast_db=rast_out_db, diff_attrs='all') #C,H,W,3
            N, Nb = dr.interpolate(v_nrm, rast_out, v_idx, rast_db=rast_out_db, diff_attrs='all') #C,H,W,3
            final_normal = T * normal[..., 0:1] - B * normal[..., 1:2] + N * torch.clamp(normal[..., 2:3], min=0.0)
            final_normal = tfunc.normalize(final_normal, dim=-1)
        else:
            final_normal = normal

        color = final_normal
        alpha = torch.clamp(rast_out[..., -1:], max=1) #C,H,W,1
        color = torch.concat((color,alpha),dim=-1) #C,H,W,4
        color = dr.antialias(color, rast_out, vertices_clip, v_idx) #C,H,W,4
        return color #C,H,W,4
