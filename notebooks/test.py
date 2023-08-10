import numpy as np

img = np.load("/home/nics/Work/sp-sold2/assets/img/img_origin.npy")
warped_img = np.load("/home/nics/Work/sp-sold2/assets/img/img_warped.npy")
pts = np.load("/home/nics/Work/sp-sold2/assets/img/points_origin.npy")
warped_pts = np.load("/home/nics/Work/sp-sold2/assets/img/points_warped.npy")
h = np.load("/home/nics/Work/sp-sold2/assets/img/homo.npy")
from utils.superpoint import SuperPointFrontend
from skimage import color 

weights_path = "/home/nics/Work/SuperPointPretrainedNetwork/superpoint_v1.pth"
nms_dist = 4
conf_thresh = 0.015
nn_thresh = 0.7
cuda = True
fe = SuperPointFrontend(weights_path=weights_path,
                          nms_dist=nms_dist,
                          conf_thresh=conf_thresh,
                          nn_thresh=nn_thresh,
                          cuda=cuda)
grayim1 = color.rgb2gray(img).astype(np.float32)
grayim2 = warped_img/255.
# print(grayim2.shape)
# print(grayim.shape)
pts1, desc1, _ = fe.run(grayim1)
pts2, desc2, _ = fe.run(grayim2)
print(pts2.shape, desc2.shape)