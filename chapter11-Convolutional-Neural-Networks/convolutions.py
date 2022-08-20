from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    #grab the spatial dimensions of the image and kernal
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    #allocate memory for the output image, taking care to "pad"
    #the borders of the input image so the spatial size are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")
    #loop over the input image "sliding " the kernel cross 
    #each (x,y) coordinate from left to right ,up to bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplication between the ROI and
            # the kernel, then summing the matrix
            k = (roi * K).sum()

            # store the convolved value in the output (x, y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k
    
    # rescale the output image to be in range [0,255]
    output = rescale_intensity(output,in_range=(0,255))
    output = (output*255).astype("uint8")
    return output


#construct arguments and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
args = vars(ap.parse_args())

#construct average blurring kernels used to smooth an image
smallBlur = np.ones((7,7),dtype="float") * (1.0/ (7 * 7))
largeBlur = np.ones((21,21),dtype="float") * (1.0 / (21 * 21))

#construct a sharpening filter
sharpen = np.array(([0 ,-1, 0],[-1 ,5, -1], [0, -1 ,0]),dtype="int")

#construct the laplacian kernel to detect edges
laplacian = np.array(([0,1,0],[-2,0,2],[0,1,0]),dtype="int")
sobelx = np.array(([-1,0,1],[-1,1,1],[-1,0,1]),dtype="int")
sobely = np.array(([-1,-2,-1],[0,0,0],[1,2,1]),dtype="int")
emboss = np.array(([-2,-1,0],[-1,1,1],[0,1,2]),dtype="int")

#construct the kernel bank, a list of kernels were going to apply
# using both our custom 'convolve function and open cvs'
kernelBank = (("small_blur",smallBlur),("large_blue",largeBlur),("sharpen",sharpen),("laplacian",laplacian),("sobel_x",sobelx),
("sobel_y",sobely),
    ("emboss",emboss))

#load the input image 
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#loop over the kernals
for (kernel_n,K) in kernelBank:
    print("[INFO] applying {} kernel".format(kernel_n))
    convolveOutput =convolve(gray,K=K)
    opencvOutput = cv2.filter2D(gray,-1,K)

    #show the output image
    cv2.imshow("original",gray)
    cv2.imshow("{} -convole".format(kernel_n),convolveOutput)
    cv2.imshow("{} -opencv".format(kernel_n),opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


