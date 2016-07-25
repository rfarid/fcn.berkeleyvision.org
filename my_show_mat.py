# Reza Farid, Fugro Roames
# Created:      2016/07/15
# Last update:  2016/07/25
#
# a code to display FCNSS mat file
import os
import scipy.io
import argparse
import logging
from PIL import Image
from my_common import *
class_numbers=range(MIN_CLASS,MAX_CLASS+1)
# -------------------------------------------------------------------------
#                           main
# -------------------------------------------------------------------------
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='...')
	parser.add_argument("infile", type=str, nargs=1, help='input file')
	parser.add_argument('--as', '--use_as', type=str, dest="infile_type", default="i",
		help='type of infile: i (single-image)(*), p (image_pattern), l (image list file), v (video), ias (image-annotations), s (annotation set per entry)')
	parser.add_argument('-c', '--chosen_class_num', type=int, dest="chosen_class_num", default=CHOSEN_CLASS,
		help='just consider images which have this class number, 0 means all')
	parser.add_argument("-j", "--just_chosen_class", action="store_true", dest="save_just_chosen_class",
		help="save just chosen class")
	parser.add_argument("--log", "--loglevel", dest="loglevel",
		help="log level: DEBUG, INFO(*), WARNING, ERROR, CRITICAL")
	# -------------------------------------------------------------------    
	# Processing args
	# -------------------------------------------------------------------
	args = parser.parse_args()
	loglevel = args.loglevel
	input_file = args.infile[0]
	chosen_class_num = args.chosen_class_num
	save_just_chosen_class=args.save_just_chosen_class

	just_str=""
	if save_just_chosen_class:
		class_numbers=[chosen_class_num]
		just_str="just_"
	outfolder='0out/exp_'+extract_name(input_file)
	if chosen_class_num!=0:
		outfolder=outfolder+"_"+just_str+str(chosen_class_num)
	if not os.path.isdir(outfolder):
		os.mkdir(outfolder)

	input_file_type = args.infile_type.upper()
	infiles= load_image_names(input_file,input_file_type)
	for infile in infiles:
		mat = scipy.io.loadmat(infile)
		in_arr = mat['S']
		if chosen_class_num!=0:
			if np.count_nonzero(in_arr == chosen_class_num)==0:
				continue
		print infile
		image_path= infile.replace("SemanticLabels", "Images", 1)
		image_path= image_path.replace(".mat", ".jpg", 1)
		classes_outfile=outfolder+"/"+extract_name(image_path)+"_classes.jpg"
		im= Image.open(image_path)
		class_images=[]		
		class_images.append((im,"Original"))
		class_images.append((in_arr*COEF,"Semantic Seg"))
		class_images=highlight_classes(in_arr,class_numbers, class_images,COEF)
		save_results(classes_outfile,class_images,5,7)


