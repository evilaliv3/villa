"""Estimate an umbilicus.json (scroll core polyline) for a scroll without a published one, from a published surface
prediction: at N heights, take the largest connected component of the sheet mask, fill holes, and use the point of maximum
distance-to-boundary as the core, which is the centre of the widest lobe of the section and coincides with the winding
centre when the section is round. Control points {x,y,z,score} are written in the coordinates of level 0 OF THE STORE
you point at, and the downsample factor comes from that store's own multiscales metadata: an already-downsampled
prediction (an L2 store, say) therefore yields L2 coordinates, not scroll level 0. The store's declared voxel size is
copied into the output metadata so that this is visible rather than silent.
Usage: estimate_umbilicus.py <scroll> <surf_zarr_rel_path/> <out.json> [n_z=12] [level=3]"""
import sys, json, urllib.request, numpy as np, numcodecs
from scipy import ndimage as ndi
B = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
scroll, rel, out = sys.argv[1:4]; n_z = int(sys.argv[4]) if len(sys.argv) > 4 else 12
level = sys.argv[5] if len(sys.argv) > 5 else '3'
url = B + rel
def read_slice(level, z):
    meta = json.loads(urllib.request.urlopen(url + f'{level}/.zarray', timeout=60).read().decode())
    shape, chunks = meta['shape'], meta['chunks']; codec = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
    cz, cy, cx = chunks; iz = z // cz; outp = np.zeros((shape[1], shape[2]), np.uint8)
    for iy in range((shape[1] + cy - 1) // cy):
        for ix in range((shape[2] + cx - 1) // cx):
            try: buf = urllib.request.urlopen(url + f'{level}/{iz}/{iy}/{ix}', timeout=60).read()
            except Exception: continue
            arr = np.frombuffer(codec.decode(buf) if codec else buf, np.dtype(meta['dtype'])).reshape(chunks)
            y1, x1 = min(shape[1], (iy + 1) * cy), min(shape[2], (ix + 1) * cx)
            outp[iy * cy:y1, ix * cx:x1] = arr[z % cz, :y1 - iy * cy, :x1 - ix * cx]
    return outp, shape
def store_metadata():
    """Downsample factor of the chosen level, and the store's own voxel size, from its metadata.
    Reading the factor instead of assuming it is the rule the spiral-fitting README already states
    for lasagna_scale, and it is the difference between coordinates that are right and coordinates
    that are wrong by a power of two with no error raised."""
    f, vox = 2 ** int(level), None
    try:
        ds = json.loads(urllib.request.urlopen(url + '.zattrs', timeout=60).read().decode())['multiscales'][0]['datasets']
        f = int(round(next(d['coordinateTransformations'][0]['scale'][-1] for d in ds if str(d['path']) == level)))
    except Exception as e:
        print(f'.zattrs multiscales unusable ({e}), falling back to 2**level = {f}', flush=True)
    try:
        vox = json.loads(urllib.request.urlopen(url + 'meta.json', timeout=60).read().decode()).get('voxelsize')
    except Exception:
        pass
    return f, vox


f, store_voxelsize = store_metadata()
print(f'level {level}, scale to this store\'s level 0 = {f}, store voxel size = {store_voxelsize} um', flush=True)
meta = json.loads(urllib.request.urlopen(url + f'{level}/.zarray', timeout=60).read().decode()); Z = meta['shape'][0]
pts = []
for frac in np.linspace(0.08, 0.92, n_z):
    z = int(frac * Z); sl, shape = read_slice(level, z)
    mask = ndi.binary_closing(sl > 40, iterations=4)
    lab, n = ndi.label(mask)
    if n == 0: continue
    sizes = ndi.sum(mask, lab, range(1, n + 1)); mask = ndi.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    dist = ndi.distance_transform_edt(mask)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    pts.append({'x': int(cx * f + f // 2), 'y': int(cy * f + f // 2), 'z': int(z * f), 'score': 60})
    print(f'z={z*f}: core at x={cx*f} y={cy*f} (depth {dist.max()*f*9.4/1000:.1f} mm)', flush=True)
json.dump({'control_points': pts, 'metadata': {'source': 'estimate_umbilicus.py (max distance-to-boundary of sheet mask)', 'scroll': scroll,
           'store': rel, 'level': level, 'scale_to_store_level0': f, 'store_voxelsize_um': store_voxelsize}}, open(out, 'w'), indent=1)
print('wrote', out, len(pts), 'points')
