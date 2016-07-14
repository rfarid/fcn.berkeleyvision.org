# Reza Farid, Fugro Roames
# Created:      2016/07/13
# Last update:  2016/07/14
#
# This code is a modified version of infer.py
#
import numpy as np
from PIL import Image
# import scipy.misc
import matplotlib.pyplot as plt

import caffe
CLASS = 20
COEF = 7
dim = 500
blob_strs=['score_sem', 'score_geo'] # 'score'
sift_type='siftflow-fcn8s' #siftflow-fcn32s
partial_resize=0.50
tile_step=dim//2
max_h_perc=0.70
infolder='./0my_data'
outfolder='./0out'
infiles=[
	infolder+'/ex1_04.jpg',
	infolder+'/ex1_62.jpg',
	infolder+'/ex2_1.jpg',
	infolder+'/ex2_5.jpg',
	infolder+'/c2_f01357.jpg',
	infolder+'/c2_f01543.jpg',
	infolder+'/c2_f02392.jpg']
deploy_file=sift_type+"/deploy.prototxt"
model_file=sift_type+'-heavy.caffemodel'

def do_infer(im,net):
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	# im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
	in_ = np.array(im, dtype=np.float32)
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	in_ = in_.transpose((2,0,1))

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	# out = net.blobs['score'].data[0].argmax(axis=0)
	images=[]
	images.append([im,"Original"])

	for blob_str in blob_strs:
		# outfile=infile[:-4]+"_out_"+sift_type+"_"+blob_str+".jpg"
		out = net.blobs[blob_str].data[0].argmax(axis=0)

		if blob_str=="score_sem":
			out_class = np.copy(out)

		out = out * COEF
		images.append([out,blob_str])
		# scipy.misc.toimage(out, cmin=0, cmax=255).save(outfile)
	mr,mc=out_class.shape
	for i in range(mr):
		for j in range(mc):
			if out_class[i,j]!=CLASS:
				out_class[i,j]=0
			else:
				out_class[i,j]=1
	images.append([out_class,"Class "+str(CLASS)+"?"])
	return images

def save_results(outfile, images):
	plt.figure(1)
	num_images=len(images)
	for i in range(num_images):
	    plt.subplot(2,2,i+1), plt.imshow(images[i][0])
	    plt.xticks([]),plt.yticks([])
	    plt.title(images[i][1])
	plt.savefig(outfile)
	# plt.show()

def form_boundary(left_top_corner,bb_max,tile):
	st_w,st_h=left_top_corner
	end_h=st_h+tile
	end_w=st_w+tile
	if end_h>bb_max[1]:
		end_h=bb_max[1]
		st_h=bb_max[1]-tile
	if end_w>bb_max[0]:
		end_w=bb_max[0]
		st_w=bb_max[0]-tile
	return (st_w,st_h,end_w,end_h)

# -------------------------------------------------------------------------
#                           main
# -------------------------------------------------------------------------
if __name__ == '__main__':

	# load net
	# net = caffe.Net('fcn8s/deploy.prototxt', 'fcn8s/fcn8s-heavy-40k.caffemodel', caffe.TEST)
	net = caffe.Net(deploy_file, model_file, caffe.TEST)

	for infile in infiles:
		print "Input:",infile
		index=infile.rfind("/")
		filename=infile[index+1:-4]
		outfile_prefix = outfolder+"/"+filename + "_out_" + sift_type
		im = Image.open(infile)
		w,h = im.size
		if w>dim or h>dim:
			input_images=[]			
			print "w=",w,"h=",h, "/\\" * 30
			# full resize - keep aspect ratio
			im.thumbnail((dim,dim), Image.ANTIALIAS)
			input_images.append((im,outfile_prefix+"_thumb.jpg"))
			im = Image.open(infile)
			# full resize - do not keep aspect ratio
			im_full_resized = im.resize((dim,dim), Image.ANTIALIAS)
			input_images.append((im_full_resized,outfile_prefix+"_resized_"+str(dim)+"x"+str(dim)+".jpg"))
			new_w=int(float(w)*partial_resize)
			new_h=int(float(h)*partial_resize)
			print "new::w=",new_w,"h=",new_h, "+" * 30
			im = im.resize((new_w,new_h),Image.ANTIALIAS)
			new_h = int(float(new_h) * max_h_perc)
			counter=1
			prev_box=(0,0,0,0)
			for j in range(0,new_h-tile_step//2, tile_step):
				for i in range(0,new_w-tile_step//2, tile_step):
					box=form_boundary((i,j),(new_w,new_h),dim)
					if box!=prev_box:
						# print i,j,box
						region=im.crop(box)
						input_images.append((region,outfile_prefix+"_tile_"+str(counter)+".jpg"))
						counter+=1
						prev_box=box
		else:
			input_images=[(im,outfile_prefix+".jpg")]
		counter=0
		for input_image in input_images:
			print "Output:", input_image[1]+"..."
			images = do_infer(input_image[0],net)
			save_results(input_image[1],images)