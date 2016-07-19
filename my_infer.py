# Reza Farid, Fugro Roames
# Created:      2016/07/13
# Last update:  2016/07/18
#
# This code is a modified version of infer.py
#
import os
import caffe
import logging
import argparse
import numpy as np
from PIL import Image
from my_common import *

dim = 500
# blob_strs=['score_sem', 'score_geo'] # 'score'
# nrows=2
# ncols=2
blob_strs=['score_sem'] # 'score'
nrows=1
ncols=3
sift_type='siftflow-fcn8s' #siftflow-fcn32s
partial_resize=0.50
tile_step=dim//2
max_h_perc=0.70
infolder='./0my_data'
outfolder_prefix='./0out/exp'
deploy_file=sift_type+"/deploy.prototxt"
model_file=sift_type+'-heavy.caffemodel'
LOG_LEVEL_DEF = logging.INFO
class_numbers=range(MIN_CLASS,MAX_CLASS+1)
apply_filter=False
DO_TILING=False
DO_RESIZE=False
# -------------------------------------------------------------------------
#                               functions
# -------------------------------------------------------------------------

def do_infer(im,net):
	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	# im = Image.open('pascal/VOC2010/JPEGImages/2007_000129.jpg')
	if apply_filter:
		in_ = np.array(im, dtype=np.float32)
		in_ = apply_emboss_filter(in_)
	else:
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
			out_semantic = np.copy(out)

		out = out * COEF
		images.append([out,blob_str])
		# scipy.misc.toimage(out, cmin=0, cmax=255).save(outfile)
	out_class_chosen = highlight_class(out_semantic, chosen_class_num,COEF)
	images.append([out_class_chosen,"Class "+str(chosen_class_num)+"?"])

	return out_semantic, images

def load_and_infer(infiles, save_classes_separate, outfolder):
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    else:
    	print "exists"
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
			if DO_RESIZE:
				# full resize - do not keep aspect ratio
				im_full_resized = im.resize((dim,dim), Image.ANTIALIAS)
				input_images.append((im_full_resized,outfile_prefix+"_resized_"+str(dim)+"x"+str(dim)+".jpg"))
			if DO_TILING:
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
			# if w<dim and h<dim:
			# 	im.thumbnail((dim,dim), Image.ANTIALIAS)
			input_images=[(im,outfile_prefix+".jpg")]
		counter=0
		for input_image in input_images:
			print "Output:", input_image[1]+"..."
			out_semantic, images = do_infer(input_image[0],net)
			save_results(input_image[1],images,nrows,ncols)
			if save_classes_separate:
				classes_outfile=input_image[1][:-4]+"_classes.jpg"
				class_images=[]
				class_images.append(images[0][:])
				class_images.append(images[1][:])
				class_images_out=highlight_classes(out_semantic, class_numbers, class_images, COEF)
				# for class_number in class_numbers:
				# 	out_class_temp = highlight_class(out_semantic,class_number,COEF)
				# 	class_images.append((out_class_temp,class_names[class_number-1]))
				# save_results(classes_outfile,class_images_out,5,7)
				advanced_save_results(classes_outfile,class_images_out)

def choose_experiment_and_infer():
	infiles=[
		infolder+'/ex1_04.jpg',
		infolder+'/ex1_62.jpg',
		infolder+'/ex2_1.jpg',
		infolder+'/ex2_5.jpg',
		infolder+'/c2_f01357.jpg',
		infolder+'/c2_f01543.jpg',
		infolder+'/c2_f02392.jpg']
	num_files=len(infiles)
	experiments = ["No separate classes","Separate Classes","Just Pole examples"]
	save_classes_separate_option=[False, True, True]
	n=len(experiments)
	chosen_indices=[range(num_files),range(num_files),range(4)]
	print chosen_indices
	for i in range(n):
		print str(i+1)+". "+experiments[i]
	n = raw_input("Experiment number? ")
	if len(n)<1 or int(n)<1 or int(n)>len(experiments):
		print "quit."
		quit()
	n = int(n)
	chosen_files = [infiles[i] for i in chosen_indices[n-1]]
	print chosen_files
	return chosen_files, save_classes_separate_option[n-1],n


# -------------------------------------------------------------------------
#                           main
# -------------------------------------------------------------------------
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='load files and run infer using fcnss')
	parser.add_argument("-in","--infile", type=str, dest="infile", default=None, help='input file')
	parser.add_argument('--as', '--use_as', type=str, dest="infile_type", default="i",
	                    help='type of infile: i (single-image)(*), p (image_pattern), l (image list file), v (video), ias (image-annotations), s (annotation set per entry)')
	parser.add_argument("-s", "--save_classes_separate", action="store_true", dest="save_classes_separate",
		help="save the result for each class separately")
	parser.add_argument('-f', "--apply_emboss_filter", action="store_true", help="apply emboss filter on images")
	parser.add_argument('-c', '--chosen_class_num', type=int, dest="chosen_class_num", default=CHOSEN_CLASS,
	                    help='just consider images which have this class number, 0 means all')
	parser.add_argument("-j", "--just_chosen_class", action="store_true", dest="save_just_chosen_class",
		help="save just chosen class")
	parser.add_argument("--log", "--loglevel", dest="loglevel",
	                    help="log level: DEBUG, INFO(*), WARNING, ERROR, CRITICAL")
	parser.add_argument('--tile', "--apply_tiling", dest="apply_tiling", action="store_true", help="apply tiling")
	parser.add_argument('--rnrk', "--resize_not_ratio_kept", dest="resize_not_ratio_kept", action="store_true", help="resize without keeping the ratio")
	# -------------------------------------------------------------------
	# Processing args
	# -------------------------------------------------------------------
	args = parser.parse_args()
	loglevel = args.loglevel
	input_file = args.infile
	input_file_type = args.infile_type.upper()
	set_log_config(loglevel, LOG_LEVEL_DEF)
	save_classes_separate=args.save_classes_separate
	apply_filter=args.apply_emboss_filter
	chosen_class_num = args.chosen_class_num
	save_just_chosen_class=args.save_just_chosen_class
	DO_TILING=args.apply_tiling
	DO_RESIZE=args.resize_not_ratio_kept

	just_str=""
	if save_just_chosen_class:
		class_numbers=[chosen_class_num]
		just_str="_just_"+str(chosen_class_num)
	# load net
	# net = caffe.Net('fcn8s/deploy.prototxt', 'fcn8s/fcn8s-heavy-40k.caffemodel', caffe.TEST)
	net = caffe.Net(deploy_file, model_file, caffe.TEST)

	if input_file is None:
		infiles,save_classes_separate,n=choose_experiment_and_infer()
		load_and_infer(infiles,save_classes_separate,outfolder_prefix+str(n)+just_str)
	else:
		input_files= load_image_names(input_file,input_file_type)
		index=input_file.rfind("/")
		exp_name=input_file[index+1:]
		index=exp_name.rfind(".")
		if index>-1:
			exp_name=exp_name[:index]
		load_and_infer(input_files, save_classes_separate, outfolder_prefix+"_"+exp_name+just_str)



