import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import class_VConvFilter
FIG_SIZE_W=12
FIG_SIZE_H=10

CHOSEN_CLASS = 20
COEF = 7
MAX_CLASS = 33
MIN_CLASS = 1
class_names=['awning', 'balcony', 'bird', 'boat', 'bridge',
'building', 'bus', 'car', 'cow', 'crosswalk',
'desert', 'door', 'fence', 'field', 'grass',
'moon','mountain','person','plant','pole',
'river', 'road', 'rock', 'sand', 'sea',
'sidewalk', 'sign', 'sky', 'staircase', 'streetlight',
'sun', 'tree', 'window']

def set_log_config(loglevel, default_log_level):
    """Sets the logging config

    Args:
        loglevel (string): desired loglevel
        default_log_level: default value for loglevel in case of None input
    """
    if loglevel is not None:
        numeric_level = getattr(logging, loglevel.upper(), None)
        if isinstance(numeric_level, int):
            logging.basicConfig(level=numeric_level, format='%(asctime)s %(message)s')
    else:
        logging.basicConfig(level=default_log_level, format='%(asctime)s %(message)s')

def read_column_from_file(infile,column_num=1,sep=None,comment_str="#",verbose=False):
    handle = open(infile)
    if verbose:
        print "Processing ",infile,"..."
    output=[]
    column_num-=1
    for line in handle:
        line = line.rstrip()
        if line.startswith(comment_str):
            continue
        columns=line.split(sep)
        # print columns
        try:
            value=columns[column_num]
            output.append(value)
        except:
            pass            
    handle.close()
    if verbose:    
        print "Done."
    return output

def load_image_names(input_str, input_type=None):
    if input_type is None:
        return [input_str]
    input_type = input_type.upper()
    if input_type == 'I':
        return [input_str]
    elif input_type == 'P':
        logging.debug("reading files using the provided pattern...")
        image_names = glob.glob(input_str)
    elif input_type == 'L':
        logging.debug("reading file names from the provided list...")
        image_names = read_column_from_file(input_str,1)
    else:
        logging.warning("Unknown type for loading image names")
        image_names = None
    return image_names

def highlight_classes(in_array, class_numbers, class_images, coef):
    for class_number in class_numbers:
        out_class_temp = highlight_class(in_array,class_number,coef)
        class_images.append((out_class_temp,class_names[class_number-1]))
    return class_images

def highlight_class(in_array, class_num, coef=1):
    out_array = np.copy(in_array)
    mr,mc=out_array.shape
    for i in range(mr):
        for j in range(mc):
            if out_array[i,j]!=class_num:
                out_array[i,j]=0
    out_array = out_array * coef
    return out_array

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

def save_results(outfile, images,nrows=2,ncols=2,do_show=False):
    # plt.figure(1)
    plt.figure(1,figsize=(FIG_SIZE_W,FIG_SIZE_H))
    num_images=len(images)
    for i in range(num_images):
        plt.subplot(nrows,ncols,i+1), plt.imshow(images[i][0])
        plt.xticks([]),plt.yticks([])
        plt.title(images[i][1],fontsize=10)
    if outfile!="":
        plt.savefig(outfile, bbox_inches='tight')
    else:
        do_show=True
    if do_show:
        plt.show()

def advanced_save_results(outfile, images,nrows=8,ncols=8,do_show=False,trows=4,tcols=4,num_t=2):
    plt.figure(1,figsize=(FIG_SIZE_W,FIG_SIZE_H))
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    G = gridspec.GridSpec(nrows, ncols)
    num_images=len(images)
    counter = 1
    crow=trows-1
    ccol=0
    for i in range(num_images):
        if i<num_t:
            axest=plt.subplot(G[0:trows-1, i*tcols:i*tcols+tcols])
            axest.set_title(images[i][1],fontsize=10)
            axest.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            axest.imshow(images[i][0])
        else:
            axes=plt.subplot(G[crow, ccol])
            axes.set_title(images[i][1],fontsize=10)
            axes.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            axes.imshow(images[i][0])
            ccol+=1
            if ccol>=ncols:
                ccol=0
                crow+=1
    if outfile!="":
        plt.savefig(outfile, bbox_inches='tight')
    else:
        do_show=True
    if do_show:
        plt.show()

def extract_name(infile_path):
    index=infile_path.rfind("/")
    out_name=infile_path[index+1:]
    index=out_name.rfind(".")
    if index>-1:
        out_name=out_name[:index]
    return out_name

def apply_emboss_filter(io_image):
    """Applies Emboss filter to the input image
    
    Args:
        io_image (image)    

    Returns:
        io_image (image)
    """
    vconv = class_VConvFilter.EmbossFilter()
    vconv.apply(io_image, io_image)
    return io_image
