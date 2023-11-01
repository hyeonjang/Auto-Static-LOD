from lib import func, geometry, render, texture
import numpy as np
import torch
from tqdm import tqdm

param = {
    "target" : "./data/John_the_Baptist.obj",
    "resolution" : [512, 512],
    "geometry" : {
        "iter" : 200,
        "lr" : 2e-2,
        "max_vertices" : 1000,
    },
    "texture" : {
        "iter" : 300,
        "lr" : 2e-2,
        "lr_ramp" : 1e-1,
    },
}

def optimize_mesh(renderer, ref, param):

    # (ref_v, ref_f, ref_n, n_idx, ref_t, ref_t_idx) = func.load_obj_(param["target"])
    (ref_v, ref_f, ref_n, _, _, _) = ref
    (opt_v, opt_f, _, _, opt_t, opt_t_idx) = func.load_obj("data/disk.obj")
    
    ref_v = geometry.normalize_vertices(ref_v)
    ref_n = geometry.calc_vertex_normals(ref_v, ref_f.type(torch.long))
    ref_img = renderer.render_normal(ref_v, ref_f, ref_n, ref_f)

    opt_v = geometry.normalize_vertices(opt_v)

    optim = geometry.MeshOptimizer(opt_v, opt_f.type(torch.long), max_vertices=param["max_vertices"])
    opt_v = optim.vertices

    for i in range(param["iter"] + 1):

        optim.zero_grad()
        opt_n = geometry.calc_vertex_normals(opt_v, opt_f.type(torch.long))
        opt_img = renderer.render_normal(opt_v, opt_f.type(torch.int32), opt_n, opt_f.type(torch.int32))
        
        loss = torch.mean((ref_img - opt_img).abs()) # L2 pixel loss.
        loss.backward()
        optim.step()

        opt_v, opt_f = optim.remesh()
        pbar.update(1)

    opt_t = geometry.unwrap(opt_v.detach().cpu().numpy(), opt_f.detach().cpu().numpy())
    opt_t, opt_t_idx = torch.tensor(opt_t, device='cuda'), opt_f.type(torch.int32)
    return (opt_v.detach(), opt_f, _, _, opt_t, opt_t_idx)

def optimize_texture_normal(renderer, ref, opt, param):

    ref_v, ref_f, ref_n, _, ref_tex, ref_tex_idx = ref
    ref_v = geometry.normalize_vertices(ref_v) # more think the normalization method, currently unit sphere
    ref_n = geometry.calc_vertex_normals(ref_v, ref_f.type(dtype=torch.long))

    with torch.no_grad():
        ref_texture = { 
            "albedo" : texture.create(np.array([0, 0, 0.5]), [1024, 1024], False)            ,
            "normal" : texture.create(np.array([0, 0, 0.5]), [1024, 1024], False)
        }

        if ref_tex is None:
            ref_img = renderer.render_normal(ref_v, ref_f, ref_n, ref_f)
        else:
            ref_img = renderer.render_texture_normal(ref_v, ref_f, ref_n, ref_f, ref_tex, ref_tex_idx, ref_texture)
        
    v, v_idx, n, _, t, t_idx = opt
    v = geometry.normalize_vertices(v)
    n = geometry.calc_vertex_normals(v, v_idx.type(dtype=torch.long))

    opt_texture = { 
        "albedo" : torch.full([1024, 1024, 3], 0.2, device='cuda', requires_grad=False),
        "normal" : torch.full([1024, 1024, 3], 0.1, device='cuda', requires_grad=True),
    }    

    optimizer = torch.optim.Adam([opt_texture["normal"]], lr=param["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: param["lr_ramp"]**(float(x)/float(param["iter"])))
    for i in range(param["iter"] + 1):

        opt_img = renderer.render_texture_normal(v, v_idx, n, v_idx, t, t_idx, opt_texture)

        loss = torch.mean((ref_img - opt_img)**2) # L2 pixel loss.
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.update(1)

    return opt_texture

def optimize():

    global pbar
    pbar = tqdm(total=param["geometry"]["iter"] + param["texture"]["iter"], desc="Task progress", unit="units")
    
    mv, proj = func.make_star_cameras(7, 7, image_size=param["resolution"])
    renderer = render.Rasterizer(mv, proj, param["resolution"])
    # return
    ref = func.load_obj(param["target"])
    opt = optimize_mesh(renderer, ref, param["geometry"])
    tex = optimize_texture_normal(renderer, ref, opt, param["texture"])
    
    pbar.close()

    # save
    func.save_obj(opt[0], opt[1], opt[4], opt[5], f"result_{param['geometry']['max_vertices']}.obj")
    # texture.save_image(f"result_{param['geometry']['max_vertices']}.png", ((tex["normal"]+1)*0.5).detach().cpu().numpy())
    texture.save_image(f"result_{param['geometry']['max_vertices']}.png", ((tex["normal"]+1)*0.5).detach().cpu().numpy())
    return

# real run
if __name__ == "__main__" :
    optimize()