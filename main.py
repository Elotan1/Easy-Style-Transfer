# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


"""Generate style mixing image matrix using pretrained network pickle."""
from flask import Flask, render_template, request, session
import shutil
import re
from pathlib import Path
from typing import List
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy


#----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'super_secret'
has_cleared = False

@app.before_request
def sess_var():
    global has_cleared
    if not has_cleared:
        session.clear()
        has_cleared = True
    if 'sty_seed' not in session:
        session['sty_seed'] = None
    if 'rsd' not in session:
        session['rsd'] = None

@app.route("/")
def home():
    session.clear()
    return render_template("index.html")

@app.route("/style")
def style():
    return render_template("page2.html", rsd=None, sty_seed=None, images=[])

@app.route("/training")
def training():
    return render_template("training.html")

@app.route("/background")
def background():
    return render_template("background.html")

@app.route("/method")
def method():
    return render_template("method.html")

@app.route("/about_us")
def about_us():
    return render_template("about_us.html")

@app.route("/sty", methods=["POST"])
def sty():
    path = Path("./static/Generated")
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  
            elif item.is_dir():
                shutil.rmtree(item)  

    session['sty_seed'] = request.form.get("style")  
    rsd = session.get('rsd')  
    sty_seed = session.get('sty_seed')
    
    try:
        if rsd is not None and (sty_seed is not None and int(sty_seed) != 0):
            generate_style_mix(int(rsd), int(sty_seed))
            print("generate_style_mix executed")
            return render_template('page2.html', rsd=rsd, sty_seed=sty_seed, images=[f"{rsd}-{sty_seed}.png", f"{rsd}-{rsd}.png", f"{sty_seed}-{sty_seed}.png"])
    except Exception as e:
        print(f"Error in /sty route: {e}")
        return f"An error occurred: {str(e)}", 500
    return "", 204

@app.route("/random", methods=["POST"])
def random():
    path = Path("./static/Generated")
    if path.exists() and path.is_dir():
        for item in path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()  
            elif item.is_dir():
                shutil.rmtree(item)  

    session['rsd'] = request.form.get("rand") 
    rsd = session.get('rsd')  
    sty_seed = session.get('sty_seed')  
    
    try:
        if rsd is not None and (sty_seed is not None and int(sty_seed) != 0):
            generate_style_mix(int(rsd), int(sty_seed))
            print("generate_style_mix executed")
            return render_template('page2.html', rsd=rsd, sty_seed=sty_seed, images=[f"{rsd}-{sty_seed}.png", f"{rsd}-{rsd}.png", f"{sty_seed}-{sty_seed}.png"])
    except Exception as e:
        print(f"Error in /random route: {e}")
        return f"An error occurred: {str(e)}", 500
    return "", 204
#----------------------------------------------------------------------------


network_pkl = 'network-snapshot-003000.pkl'
outdir = './static/Generated'
truncation_psi = 1
noise_mode = 'const'
col_styles = [0,1,2,3,4,5,6]


#----------------------------------------------------------------------------


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''


    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


#----------------------------------------------------------------------------


def generate_style_mix(image_array = None, style= None):
    """Generate images using pretrained network pickle.


    Examples:


    \b
    python style_mixing.py --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    row_seeds = [image_array]
    col_seeds = [style]


    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}


    print('Generating images...')
    all_images = G.synthesis(all_w, noise_mode=noise_mode)
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}


    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].clone()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = G.synthesis(w[np.newaxis], noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[(row_seed, col_seed)] = image[0].cpu().numpy()


    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')


    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + row_seeds):
        for col_idx, col_seed in enumerate([0] + col_seeds):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(f'{outdir}/grid.png')


#----------------------------------------------------------------------------


if __name__ == "__main__":
    app.run()
