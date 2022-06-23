import numpy as np
import string
from PIL import ImageDraw, ImageFont, Image
from skimage import io
import cv2
import os
import pdb

def add_text(image_path):
    folder = os.path.dirname(img_path)
    image = Image.open(image_path)
    #img = Image.fromarray(img)
    ascii_chars = np.array(list((string.ascii_uppercase + string.ascii_lowercase + string.digits)))
    font = ImageFont.truetype("arial.ttf", np.random.randint(16, 32))
    x = np.random.randint(0, image.width / 2)
    y = np.random.randint(0, image.height)
    strlen = np.random.randint(5, 15)
    text = "".join(np.random.choice(ascii_chars, strlen))
    ImageDraw.Draw(image).text(
       (x, y), text, fill=np.random.randint(0, 256), font=font)
    #img = np.array(img)
    image.save(folder+"\\text.ppm")

def add_watermark(image_path, watermark_path):
    folder = os.path.dirname(img_path)
    image = Image.open(image_path)
    watermark = Image.open(watermark_path)
    watermark = watermark.resize((image.size[0],image.size[1]), Image.ANTIALIAS) 
    image.paste(watermark, (0, 0), watermark)
    image.save(folder + "\\img_watermark.ppm")

def resize_image(image_path):
    folder = os.path.dirname(img_path)
    image = cv2.imread(image_path)
    height, width,w = image.shape
    image_resized = cv2.resize(image, (round(np.random.randint(width / 4, width / 2)), round(np.random.randint(height / 4, height / 2))),interpolation = cv2.INTER_AREA) 
    cv2.imwrite(folder+"\\resized.ppm", image_resized)

def rotate_image(image_path, angle):
    folder = os.path.dirname(img_path)
    image = cv2.imread(image_path)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    img_rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    cv2.imwrite(folder + "\\rotated_"+str(angle)+".ppm", img_rotated) 

def add_noise(image_path):
    folder = os.path.dirname(img_path)
    image = cv2.imread(image_path)
    img_zg = image.copy()
    zg = np.random.normal(25, 3, [image.shape[0], image.shape[1], image.shape[2]])
    img_zg = image + zg
    cv2.imwrite(folder + "\\noise.ppm", img_zg) 

def add_filter(image_path):
    folder = os.path.dirname(img_path)
    image = cv2.imread(image_path)
    filter1 = np.full(image.shape, (0,125,150), np.uint8)
    fused_img  = cv2.addWeighted(image, 0.8, filter1, 0.2, 0)
    cv2.imwrite(folder + '\\filtered.ppm', fused_img) 

def modify_compression(image_path):
    folder = os.path.dirname(img_path)
    image = cv2.imread(image_path)
    cv2.imwrite(folder + '\\compressed.ppm', image,  [cv2.IMWRITE_JPEG_QUALITY, 9])

def crop_image(image_path, percent):
    folder = os.path.dirname(img_path)
    image = cv2.imread(image_path)
    h, w, k = image.shape
    croped_img = image[int(h*percent/100):int(h*(100-percent)/100), int(w*percent/100):int(w*(100-percent))]
    cv2.imwrite(folder + "\\cropped_"+str(angle)+".ppm", croped_img)

def vertical_flip(image_path):
    folder = os.path.dirname(img_path)
    img = io.imread(image_path)
    img_vflip = img[:, ::-1]
    io.imsave(folder+"\\img_vflip.ppm", img_vflip)

def horizontal_flip(image_path):
    folder = os.path.dirname(img_path)
    img = io.imread(image_path)
    img_hflip = img[::-1, :]
    io.imsave(folder+"\\img_hflip.ppm", img_hflip)


database_path = 'E:\\Master An2\\disertatie\\proiect\\database\\hpatches-sequences-release'
directories = [ os.path.basename(f.path) for f in os.scandir(database_path) if f.is_dir() ]


for i, folder in enumerate(directories) :
    img_path = database_path + "\\"+ folder + "\\1.ppm"
    watermark_path = 'watermark.png'

    add_text(img_path)
    add_watermark(img_path,watermark_path)
    resize_image(img_path)
    for angle in range(5,25,5):
        rotate_image(img_path, angle)
        crop_image(img_path, angle)
    add_noise(img_path)
    add_filter(img_path)
    modify_compression(img_path)

