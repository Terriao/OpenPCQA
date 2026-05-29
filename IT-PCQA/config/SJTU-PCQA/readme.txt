********************Predicting the Perceptual Quality of Point Cloud: A 3D-to-2D Projection-Based Exploration*******************************

-------------------Qi Yang, Zhan Ma, Senior Member, IEEE, Yiling Xu, Member, IEEE, Rongjun Tang, and Jun Sun-----------------------------

This database made by Qi Yang(yang_littleiqi@sjtu.edu.cn) and Rongjun Tang(thekey@sjtu.edu.cn) from Shanghai Jiao Tong University, we 
welcome everyone to carry on the test and propose the modification opinion. If you use our database in your paper, please cite our paper:
Predicting the Perceptual Quality of Point Cloud: A 3D-to-2D Projection-Based Exploration, submitted to Trans. Multimedia

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
There are 9 reference samples, including redandblack, loot, soldier, longdress, Hhi, shiva, statue, ULB Unicorn and Romanoillamp, each sample 
was processed into 7 different distortion in six levels:

number            distortion type
  0-5            OT:	 Octree-based compression
  6-11          CN:	 Color Noise
12-17          DS:	 Downscaling
18-23	   D+C:	 Downscaling and Color noise
24-29	   D+G:	 Downscaling and Geometry Gaussian noise
30-35	   GGN:	 Geometry Gaussian noise
36-41	   C+G:	 Color noise and Geometry Gaussian noise


***********************************************Parameters setting******************************************************
OT:  Compression noise is exemplified using the octree pruning method provided in well-known Point Cloud
Library (PCL) (http://pointclouds.org/downloads/). Octree pruning removes leaf nodes to adjust tree resolution
for compression purpose. Here, we have experimented different compression levels by removing points at 13%,
27%, 43%, 58%, 70% and 85%. It is difficult to guarantee the point removal percentage at the exact number.
Thus we allow ±3% deviation.

CN: Color noise, or photometric noise is applied to the photometric attributes (RGB values) of the points. We
inject the noise for 10%, 30%, 40%, 50%, 60%, and 70% points that are randomly selected, where noise levels are
respectively and again randomly given within ±10, ±30, ±40, ±50, ±60, and ±70 for corresponding points (e.g.,
10% random points with ±10 noise, 30% random points with ±30, and so on so forth). Noise is equally applied to
R, G, B attributes. Clipping is used if the noisy intensity p = p + n, is out of the range of [0, 255], e.g., if p <  0,
p=  0; and if p > 255, p = 255.


DS: We randomly downsample the point clouds by removing 15%, 30%, 45%, 60%, 75%, 90% points
from the original point clouds. We directly utilize the downscaling function pcdownscample() offered by
the Matlab software.

DS+CN or D+C: We combine aforementioned DS and CN where the downsampling process is firstly applied
and then the color noise is added in a consecutive order, e.g., 15% DS and 10% random points with ±10 noise,
30% DS and 30% random points with ±30 noise, and so on so forth).

DS+GGN or D+G: GGN and DS are superimposed. The DS process is firstly applied before augmenting the
GGN consecutively, e.g., 15% DS with 0:05% GGN , 30% DS with 0:1% GGN, and so on so forth).


GGN: We apply Gaussian distributed geometric shift to each point randomly. In this study, all the points will be
augmented with a random geometric shift that is within 0:05%, 0:1%, 0:02%, 0:5%, 0:7%, 1:2% of the bounding
box.

CN+GGN or C+G: Both GGN and CN are superimposed. The GGN is firstly applied , and the is the CN, 
e.g., 0:05% GGN and 10% random points with ±10 noise , 0:1% GGN and 30% random points with ±30
noise, and so on so forth).

************************************************************************************************************************
mos: Final_MOS.mat
42*9 matrix
9 represents 9 samples, the order is ['redandblack','UBL_unicorn','loot','soldier','Romanoillamp','longdress','statue','shiva','hhi']
42 represents 42 distortions
  1-6            OT:	 Octree-based compression
  7-12          CN:	 Color Noise
13-18          DS:	 Downscaling
19-24	   D+C:	 Downscaling and Color noise
25-30	   D+G:	 Downscaling and Geometry Gaussian noise
31-36	   GGN:	 Geometry Gaussian noise
37-42	   C+G:	 Color noise and Geometry Gaussian noise