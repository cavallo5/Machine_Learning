'''
usage: python create_dataset.py INFILE.tif PREFIX 15

convert input image to grayscale, add noise, save noisy image, create txt files with original image and noisy image

'''

from PIL import Image
import numpy as np
import sys

if len(sys.argv) != 4:
	print ("argument error:   python", sys.argv[0], " input_image.tif prefix noiselevel")
	sys.exit()
	

INFILENAME = sys.argv[1]
PREFIX = sys.argv[2]
NOISE_LEVEL = int(sys.argv[3])


OUTFILENAMECLEAN = PREFIX + '-greyscale.bmp'
OUTFILENAMENOISE = PREFIX + '-greyscale-noise-'+str(NOISE_LEVEL)+'.bmp'
CLEANDATA = PREFIX + "-data-clean.txt"
NOISEDATA = PREFIX + "-data-noise-" + str(NOISE_LEVEL) + ".txt"
    


print ("loading image", INFILENAME, "and converting to grayscale...")
img = Image.open(INFILENAME).convert('L')
img.save(OUTFILENAMECLEAN)

bnarray = np.array(img)
SIZE =  bnarray.shape
print ("img shape: ", SIZE,) 
print ("min:", bnarray.min(), "max:", bnarray.max())

print ("generating random noise...")
noise = np.random.randint(-NOISE_LEVEL,NOISE_LEVEL+1,img.size)

bnarray_noise = bnarray+noise
#fixing limits
print ("fixing limits...")
print ("\tORI min=%d max=%d" % (bnarray.min(),bnarray.max()))
print ("\tPRE min=%d max=%d" % (bnarray_noise.min(),bnarray_noise.max()))
if bnarray_noise.min() < 0 or bnarray_noise.max() > 255:
    print ("\t********* reshaping preserving noise dynamic")
    bnarray_noise= 255*(bnarray_noise-bnarray_noise.min())/(bnarray_noise.max()-bnarray_noise.min())   
    #bnarray_noise[bnarray_noise > 255] = 255
    #bnarray_noise[bnarray_noise < 0] = 0
print ("\tPOST min=%d max=%d" % (bnarray_noise.min(),bnarray_noise.max()))

print ("saving image with noise to", OUTFILENAMENOISE, "...")
img_noise = Image.fromarray(np.uint8(bnarray_noise))
img_noise.save(OUTFILENAMENOISE)

print ("saving dataset...")
np.savetxt(NOISEDATA,np.uint8(bnarray_noise), fmt="%d")
np.savetxt(CLEANDATA,np.uint8(bnarray), fmt="%d")


