from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageFilter


main = Tk()
main.title("image processing")
main.geometry("1200x900")
main.tk_focusFollowsMouse()
main.configure(background='#658CAD')
fr = Frame(main, width="10", padx=50, pady=10, bg='#658CAD')
fr.grid(row=0, column=0)
fr1 = Frame(main, width="10", padx=50, pady=100, bg='#658CAD')
fr1.grid(row=0, column=3)

Transformations = LabelFrame(
    fr1, text='Transformations', width='10', padx=50, pady=10, bg="#658CAD")
Transformations.grid(row=0, column=1)

fillters = LabelFrame(fr1, text='Fillters', width='10', padx=50, pady=10, bg="#658CAD")
fillters.grid(row=1, column=1)

img_paths = ["images\cat.jpg", ""]

def browse_image():
    global img
    img_paths[0] = filedialog.askopenfilename()
    img = Image.open(img_paths[0])
    img = ImageTk.PhotoImage(img)
    lb['image'] = img


Browse_bt = Button(fr, command=browse_image, bg="#91c7f4", fg="black", padx=10, pady=8, width="20", height="3",
                   text="browse")
Browse_bt.pack()


# convert image to gray scale


def gray_image():
    imageGray = cv2.imread(img_paths[0], 0)
    cv2.imshow("gray scale", imageGray)
    return imageGray


gray_image_bt = Button(fr, command=gray_image, bg="#91c7f4", fg="black", padx=10, pady=8, width="20",
                       text="gray")
gray_image_bt.pack()


def Threshold():
    img = cv2.imread(img_paths[0], 0)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow('THRESH_BINARY', thresh1)
    cv2.imshow('THRESH_BINARY_INV', thresh2)
    cv2.imshow('THRESH_TRUNC', thresh3)
    cv2.imshow('THRESH_TOZERO', thresh4)
    cv2.imshow('THRESH_TOZERO_INV', thresh5)


Threshold_bt = Button(fr, command=Threshold, bg="#91c7f4", fg="black", padx=10, pady=8, width="20",
                      text="Threshold")
Threshold_bt.pack()


def save_image():
    saved_image = cv2.imread(img_paths[0])
    save_at_path = filedialog.asksaveasfilename(
        initialdir='saved_images',
        title="Save Image",
        filetypes=[("PNG Image", "*.png"),
                   ("JPEG Image", "*.jpg"), ("GIF Image", "*.gif"),
                   ("TIFF Image", "*.tiff"), ("BMP Image", "*.bmp")])
    cv2.imwrite(save_at_path, saved_image)


save_bt = Button(fr, command=save_image, bg="#91c7f4", fg="black", padx=10, pady=8, width="20", text="Save")
save_bt.pack()


# Transformations

# (1)Negative Transformation

def NegativeTransformation():
    NegativeImg = cv2.imread(img_paths[0], 0)
    height, width = np.shape(NegativeImg)
    for row in range(height):
        for col in range(width):
            NegativeImg[row][col] = 255 - NegativeImg[row][col]

    cv2.imshow("NegativeImg", NegativeImg)


Negative_bt = Button(Transformations, command=NegativeTransformation, bg="#91c7f4", fg="black", padx=10, pady=8,
                     width="20",
                     text="Negative")
Negative_bt.pack()


# (2)Log Transformation

def LogTransformation():
    LogImg = cv2.imread(img_paths[0], 0)
    h, w = np.shape(LogImg)
    for row in range(h):
        for col in range(w):
            LogImg[row][col] = int(6 * (np.log2(1 + LogImg[row][col])))
    cv2.imshow("LogTransform", LogImg)
    return LogImg


Log_bt = Button(Transformations, command=LogTransformation, bg="#91c7f4", fg="black", padx=10, pady=8, width="20",
                text="Log")
Log_bt.pack()


# (3)Power_Law Transformation
def PowerLawTransformation():
    PowLawImg = cv2.imread(img_paths[0], 0)
    h, w = np.shape(PowLawImg)
    for row in range(h):
        for col in range(w):
            #             when power decrease intensity increase
            PowLawImg[row][col] = 255 * (PowLawImg[row][col] / 255) ** 6
    cv2.imshow("PowLawTransform", PowLawImg)
    return PowLawImg


PowLaw_bt = Button(Transformations, command=PowerLawTransformation, bg="#91c7f4", fg="black", padx=10, pady=8,
                   width="20",
                   text="Power Law")
PowLaw_bt.pack()


# Contrast Stretching
# didn't work
def ContrastStretching():
    ConStrImg = cv2.imread(img_paths[0], 0) / 255
    width, height = np.shape(ConStrImg)
    for row in range(width - 1):
        for col in range(height - 1):
            if (ConStrImg[row][col] > .3) and (ConStrImg[row][col] < .7):
                ConStrImg[row][col] = ConStrImg[row][col] + .3
            else:
                ConStrImg[row][col] = ConStrImg[row][col]
    cv2.imshow("ConStrImg Slicing", ConStrImg)
    return ConStrImg


ConStr_bt = Button(fr, command=ContrastStretching, bg="#91c7f4", fg="black", padx=10, pady=8, width="20",
                   text="Contrast Stretching")
ConStr_bt.pack()


# Intensity Slicing


def IntensitySlicing():
    img1 = cv2.imread(img_paths[0], 0)
    h, w = np.shape(img1)
    sliceImg = np.zeros((h, w), dtype='uint8')
    min_range = 10
    max_range = 60
    for row in range(h):
        for col in range(w):
            if img1[row][col] > min_range and img1[row][col] < max_range:
                sliceImg[row][col] = 255
            else:
                sliceImg[row][col] = 0
    sliceImg = sliceImg / 255
    cv2.imshow("Intensity Slicing", sliceImg)
    return sliceImg


IntensitySlicing_bt = Button(
    fr, command=IntensitySlicing, text="Intensity Slicing", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
IntensitySlicing_bt.pack()


# Histogram
def HistogramCalc():
    img_array = cv2.imread(img_paths[0])
    # flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=255)
    # normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / num_pixels
    # cumulative histogram
    chistogram_array = np.cumsum(histogram_array)
    """
    STEP 2: Pixel mapping lookup table
    """
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    """
    STEP 3: Transformation
    """
    # flatten image array into 1D list
    img_list = list(img_array.flatten())
    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]
    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    cv2.imshow('After', eq_img_array)


histogram_bt = Button(fr, text="histogram", command=HistogramCalc,
                      bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
histogram_bt.pack()


# Spatial Filtaring

# (1)Mean fillter(average)
# [1,1,1]
# [1,1,1]
# [1,1,1]


def meanFillter():
    before_img = cv2.imread(img_paths[0])
    blurImg = cv2.blur(before_img, (5, 5))
    cv2.imshow('mean fillter (average) ', blurImg)


meanFillter_bt = Button(fillters, command=meanFillter, text="MeanFillter",
                        bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
meanFillter_bt.pack()


# (2)Gaussian fillter
# [1,2,1]
# [2,4,2]
# [1,2,1]


def GaussianFillter():
    before_img = cv2.imread(img_paths[0])
    gaussianImg = cv2.GaussianBlur(before_img, (5, 5), 20)
    cv2.imshow('Gaussian fillter', gaussianImg)


GaussianFillter_bt = Button(fillters, command=GaussianFillter,
                            text="GaussianFillter", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
GaussianFillter_bt.pack()


# Median Fillter


def MedianFillter():
    noisyImg = cv2.imread(img_paths[0])
    mediamFillterImg = cv2.medianBlur(noisyImg, 5)
    cv2.imshow('After Median Fillter', mediamFillterImg)


MedianFillter_bt = Button(fillters, command=MedianFillter,
                          text="MedianFillter", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
MedianFillter_bt.pack()


# Min Fillter


def MinFillter():
    noisyImg = Image.open(img_paths[0])
    Min_ = noisyImg.filter(ImageFilter.MinFilter(size=3))
    Min_.show()
    # remove solt dots


MinFillter_bt = Button(fillters, command=MinFillter, text="MinFillter", bg="#91c7f4", fg="black", padx=10, pady=8,
                       width="20")
MinFillter_bt.pack()


# Max Fillter
def MaxFillter():
    noisyImg = Image.open(img_paths[0])
    Max_ = noisyImg.filter(ImageFilter.MaxFilter(size=9))
    Max_.show()
    # remove peper dots


MaxFillter_bt = Button(fillters, command=MaxFillter, text="MaxFillter",
                       bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
MaxFillter_bt.pack()


def laplacianSharping():
    original_image = cv2.imread(img_paths[0], 0)
    laplacianImage = cv2.Laplacian(original_image, cv2.CV_64F, ksize=3)
    kernelArray = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplacianImageScratch = cv2.filter2D(src=original_image, ddepth=-1, kernel=kernelArray)
    cv2.imshow('laplacianImageScratch', laplacianImageScratch)


laplacianSharping_bt = Button(fillters, command=laplacianSharping,
                              text="laplacianSharping", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
laplacianSharping_bt.pack()


def laplacianofgaussian():
    original_image = cv2.imread(img_paths[0], 0)
    blur_image_gaussian = cv2.GaussianBlur(original_image, (3, 3), 0)
    # Second step : Apply Laplace function
    LoG_image = cv2.Laplacian(blur_image_gaussian, cv2.CV_64F, ksize=3)
    # Show Laplacian of gaussain image >> Sharpen image
    cv2.imshow('LoG_image', LoG_image)


laplacianofgaussian_bt = Button(fillters, command=laplacianofgaussian,
                                text="laplacian of gaussian", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
laplacianofgaussian_bt.pack()


def sobelSharping():
    image = cv2.imread(img_paths[0], 0)
    # Convert the image to grayscale
    # Apply the Sobel operator to the image
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Combine the horizontal and vertical gradients
    sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # Save the Sobel image
    cv2.imshow('sobel', sobel)


sobelSharping_bt = Button(fillters, command=sobelSharping,
                          text="sobelSharping", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
sobelSharping_bt.pack()


def prewitSharping():
    image = cv2.imread(img_paths[0], 0)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    # Apply the Prewitt kernels to the image
    sharp_image = cv2.filter2D(image, -1, kernel_x) + cv2.filter2D(image, -1, kernel_y)
    cv2.imshow('sharp_image', sharp_image)


prewitSharping_bt = Button(fillters, command=prewitSharping,
                           text="prewitSharping", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
prewitSharping_bt.pack()


def fourier_sharping():
    img = cv2.imread(img_paths[0], 0)
    np.seterr(divide='ignore')
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')
    plt.savefig('fft.png')


fourier_sharping_bt = Button(fr, command=fourier_sharping,
                             text="fourier_sharping", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
fourier_sharping_bt.pack()


def fourier_smoothing():
    img = cv2.imread(img_paths[0], 0)
    np.seterr(divide='ignore')
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    r = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 1

    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2, 2, 4)

    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')
    plt.savefig('fft.png')


fourier_smoothing_bt = Button(fr, command=fourier_smoothing,
                              text="fourier_smoothing", bg="#91c7f4", fg="black", padx=10, pady=8, width="20")
fourier_smoothing_bt.pack()


def Close():
    main.destroy()

exit_button_bt = Button(fr, text="Exit", command=Close, bg="#176B87", fg="white", padx=10, pady=8, width=20 )
exit_button_bt.pack()


no_image = Image.open(img_paths[0])
no_image = no_image.resize((600, 400))
no_image = ImageTk.PhotoImage(no_image)
lb = Label(main, bg="white", image=no_image)
lb.grid(row=0, column=2)
main.mainloop()
