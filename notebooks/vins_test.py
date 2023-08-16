#%%
import yaml
import cv2
import torch
from utils.vins.model import SPSOLD2ExtractModel

yamlPath = "/home/plus/Work/sp-sold2/SP-SOLD2/notebooks/config/config.yaml"
with open(yamlPath,'rb') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params["sp_sold2_model_cfg"]
model = SPSOLD2ExtractModel(params)

img1 = '/home/plus/Work/sp-sold2/SP-SOLD2/assets/0.png'
img1 = cv2.imread(img1, 0)
# img1 = cv2.resize(img1, (img1.shape[1] // scale_factor, img1.shape[0] // scale_factor),
#                   interpolation = cv2.INTER_AREA)
# img1 = (img1 / 255.).astype(float)
# torch_img1 = torch.tensor(img1, dtype=torch.float)[None, None].cuda()


lines, _ = model.extract(img1)
print(lines.shape)
#%% 
import matplotlib.pyplot as plt
def plot_lines(lines, line_colors='orange', point_colors='cyan',
               ps=4, lw=2, indices=(0, 1)):
    """Plot lines and endpoints for existing images.
    Args:
        lines: list of ndarrays of size (N, 2, 2).
        colors: string, or list of list of tuples (one for each keypoints).
        ps: size of the keypoints as float pixels.
        lw: line width as float pixels.
        indices: indices of the images to draw the matches on.
    """
    if not isinstance(line_colors, list):
        line_colors = [line_colors] * len(lines)
    if not isinstance(point_colors, list):
        point_colors = [point_colors] * len(lines)

    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) > max(indices)
    axes = [ax[i] for i in indices]
    fig.canvas.draw()

    # Plot the lines and junctions
    for a, l, lc, pc in zip(axes, lines, line_colors, point_colors):
        for i in range(len(l)):
            line = plt.lines.Line2D((l[i, 0, 0], l[i, 1, 0]),
                                           (l[i, 0, 1], l[i, 1, 1]),
                                           zorder=1, c=lc, linewidth=lw)
            a.add_line(line)
        pts = l.reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1],
                  c=pc, s=ps, linewidths=0, zorder=2)


line_seg1 = lines.cpu().numpy()
plot_lines([line_seg1], ps=3, lw=2)
plt.show()