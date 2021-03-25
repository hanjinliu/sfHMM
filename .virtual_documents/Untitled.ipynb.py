import impy as ip
ip.__version__


img = ip.imread(r"D:\Reserch\Data\210317\Qdot_4x4-positions.tif")
img = img["t=1"].affine_correction()



