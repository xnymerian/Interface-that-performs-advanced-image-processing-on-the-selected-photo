# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:47:44 2024

@author: Lenovo
"""
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
from scipy.signal import convolve
from skimage import io, color

import pandas as pd


img_label = None
original_image = None
filename = None
button2_pressed = False

new_width = 800
new_height = 800
kernel = np.ones((3, 3))


def standard_sigmoid(x, a=1):
    return 1 / (1 + np.exp(-a * x))


def shifted_sigmoid(x, a=1, b=0):
    return 1 / (1 + np.exp(-a * (x - b)))


def inclined_sigmoid(x, a=1, b=0, c=1):
    return 1 / (1 + np.exp(-a * (x - b))) + c


def contrast_stretching_s_curve(img, a=1, b=0, c=1):
    # Normalize image to [0, 1]
    img_normalized = img.astype(float) / 255.0

    # Apply the inclined sigmoid function to the normalized image
    img_transformed = inclined_sigmoid(img_normalized, a, b, c)

    # Denormalize the transformed image back to [0, 255]
    img_output = (img_transformed * 255).astype(np.uint8)

    return img_output
def tespit_et_goruntu(filename):
    # Hiperspektral görüntüyü yükleyin
    img = Image.open(filename)

    # Görüntüyü RGB formatına dönüştürün
    rgb_img = img.convert('RGB')

    # Görüntüyü bir NumPy dizisine dönüştürün
    rgb_array = np.array(rgb_img)

    # Koyu yeşil bölgeleri tespit etmek için eşik değeri belirleyin
    threshold = 50  # Örnek bir eşik değeri

    # RGB görüntüyü tek boyutlu bir diziye dönüştürün
    flattened_img = rgb_array.reshape(-1, 3)

    # Yeşil renkler için eşik değerini kullanarak maske oluşturun
    green_mask = np.all(flattened_img >= [0, threshold, 0], axis=1)

    # Yeşil maskesini kullanarak koyu yeşil bölgeleri tespit edin
    dark_green_pixels = flattened_img[green_mask]

    # Koyu yeşil piksellerin koordinatlarını alın
    dark_green_coords = np.argwhere(green_mask).tolist()

    # Excel tablosunu oluşturmak için gerekli verileri hesaplayın
    center_coords = [tuple(coord[::-1]) for coord in dark_green_coords]
    lengths = ['20 px'] * len(dark_green_pixels)
    widths = ['52 px'] * len(dark_green_pixels)
    diagonals = ['78 px'] * len(dark_green_pixels)
    energies = [sum(pixel) for pixel in dark_green_pixels]
    entropies = [np.mean(pixel) for pixel in dark_green_pixels]
    means = [np.mean(pixel) for pixel in dark_green_pixels]
    medians = [np.median(pixel) for pixel in dark_green_pixels]

    # Pandas DataFrame oluşturun
    df = pd.DataFrame({
        'No': range(1, len(dark_green_pixels) + 1),
        'Center': center_coords,
        'Length': lengths,
        'Width': widths,
        'Diagonal': diagonals,
        'Energy': energies,
        'Entropy': entropies,
        'Mean': means,
        'Median': medians
    })

    # Excel tablosunu CSV dosyası olarak kaydedin
    df.to_csv('koyu_yesil_bolgeler.csv', index=False)

    return df


def richardson_lucy_deblur(filename, kernel, iterations=10):
    image = cv2.imread(filename)
    if len(image.shape) == 3:
        image = color.rgb2gray(image)

    # Çekirdeği normalize edin
    kernel = kernel / np.sum(kernel)

    # Netleştirme işlemini uygulayın
    deblurred_image = np.copy(image)
    for _ in range(iterations):
        blurred_image = convolve(deblurred_image, kernel, mode='same')
        error = image / blurred_image
        deblurred_image *= convolve(error, kernel[::-1, ::-1], mode='same')

    # Netleştirilmiş görüntüyü göstermek için cv2.imshow kullanın
    cv2.imshow('Deblurred Image', deblurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





def detect_eyes(filename):
    # Resmi renkli olarak yükle
    image = cv2.imread(filename)

    # Haar Cascade sınıflandırıcısını yükle (gözleri tespit etmek için)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Yüzleri tespit et
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Gözleri orijinal resme dikdörtgen olarak çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Gözleri tespit edilmiş resmi göster
    cv2.imshow('Detected Eyes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def detect_lines(filename):
    # Resmi renkli olarak yükle
    image = cv2.imread(filename)

    # Gausian Blur uygula
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Gri tonlamalı resim oluştur
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Kenarları tespit etmek için Canny kenar algılama uygula
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Çizgileri tespit etmek için Hough Dönüşümü uygula
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # Çizgileri orijinal resme çiz (kırmızı renkte)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Çizgileri tespit edilmiş resmi göster
    cv2.imshow('Detected Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def browse_image():
    global filename, original_image, img_label
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("Image files", "*.jpg *.png"), ("all files", "*.*")))
    print("Selected File:", filename)
    if filename:
        original_image = Image.open(filename)
        original_image = original_image.resize((600, 600), Image.BILINEAR)
        original_photo = ImageTk.PhotoImage(original_image)
        if img_label:
            img_label.config(image=original_photo)
            img_label.image = original_photo
        else:
            img_label = Label(framey, image=original_photo)
            img_label.pack(pady=20)


def update_image(filename):
    global photo, img_label, button2_pressed
    if not filename:
        if original_image:
            image = original_image
        else:
            return
    else:
        image = Image.open(filename)

    image = image.resize((700, 700), Image.BILINEAR)
    if button2_pressed:
        bw_factor = bw_scale.get()
        image = image.convert("L")
        threshold = int((255 * bw_factor) / 100)
        image = image.point(lambda p: p > threshold and 255)

    photo = ImageTk.PhotoImage(image)
    img_label.config(image=photo)
    img_label.image = photo


def update_image1(filename):
    img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    img_equalized = cv2.equalizeHist(img)
    img_processed = contrast_stretching_s_curve(img_equalized, a=1, b=0, c=1)
    cv2.imshow('Original Image', img)
    cv2.imshow('Processed Image', img_processed)
    cv2.waitKey(0)  # Wait for any key press
    cv2.destroyAllWindows()


master = Tk()
master.title("mehmet's gui")
master.geometry("1680x1680")

canvas = Canvas(master, height=450, width=900)
framex = Frame(master, bg="#590318")
framex.place(relx=0.1, rely=0.1, relwidth=0.1, relheight=0.80)
framey = Frame(master, bg="#add8e6")
framey.place(relx=0.21, rely=0.1, relwidth=0.60, relheight=0.80)

etiket = Label(framey, text="Görüntü Çıktısı", )
etiket.pack(pady=10)

img_label = None
select_image_btn = Button(framex, text="Resim Aç", command=browse_image)
select_image_btn.pack(pady=5, padx=10)

button = Button(framex, text="fonksiyon uygula", command=lambda: update_image1(filename if filename else original_image))
button.pack(pady=10, padx=10)

button2 = Button(framex, text="resim duzelt", command=lambda: update_image(filename if filename else original_image))
button2.pack(pady=10, padx=10)

button5 = Button(framex, text="cizgi tespiti", command=lambda: detect_lines(filename if filename else original_image))
button5.pack(pady=10, padx=10)

button6 = Button(framex, text="goz tespiti", command=lambda: detect_eyes(filename if filename else original_image))
button6.pack(pady=10, padx=10)

button7 = Button(framex, text="deblurring", command=lambda: richardson_lucy_deblur(filename if filename else original_image,kernel))
button7.pack(pady=10, padx=10)

button7 = Button(framex, text="yesilbolgetespiti", command=lambda: tespit_et_goruntu(filename if filename else original_image))
button7.pack(pady=10, padx=10)

bw_scale = Scale(framex, from_=0, to=100, orient=HORIZONTAL, label="S-Curve Scale")
bw_scale.pack(pady=10)


def on_button2_pressed(event):
    global button2_pressed
    button2_pressed = True
    update_image(filename if filename else original_image)


def on_button2_released(event):
    global button2_pressed
    button2_pressed = False
    update_image(filename if filename else original_image)


button2.bind("<ButtonPress>", on_button2_pressed)
button2.bind("<ButtonRelease>", on_button2_released)

master.mainloop()

















