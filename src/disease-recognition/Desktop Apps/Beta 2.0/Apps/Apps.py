import cv2 as cv
import tkinter as tk
from tkinter.ttk import Frame, Style
import numpy as np
import pandas as pd
import os
from tkinter import filedialog as fd 
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
import tkinter.font as font
import torch
import ultralytics
from ultralytics import YOLO
import random
import string
import pickle



window = tk.Tk()

#get relatives size
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()


#print(screen_width)
#print(screen_height)


#window.state('zoomed')
window.geometry("%dx%d" % (screen_width, screen_height))
window.title("Dental Disease Detection")


result_folder = ".\\result" 
model_file = r".\Model\model.pt"

max_width = 890
max_height = 910

def open_image():
    global filename
    
    #open dialog box menu
    filename = tk.filedialog.askopenfilename(
        title="Select Dental Image",
        filetypes=[("Dental Images", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    if filename:
        filenames = filename.split("/")[-1] 
        # Update input image display
        inpt_img = cv.imread(filename)  
        inpt_img = cv.cvtColor(inpt_img, cv.COLOR_BGR2RGB)  
    
        height, width = inpt_img.shape[:2]
       
        width_ratio = float(max_width) / width
        height_ratio = float(max_height) / height
        
        resize_factor = min(width_ratio, height_ratio)

        
        width_new = int(width * resize_factor)
        height_new = int(height * resize_factor)
        img_resized = cv.resize(inpt_img, (width_new, height_new), interpolation=cv.INTER_AREA)
    
        input_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
        input_img_label.configure(image=input_img)
        input_img_label.image = input_img  # Keep reference
    
    else:
        # No image selected
        messagebox.showinfo(title="No Image Selected", message="No image was chosen.")

    
    return filename

def save_image(image):
    global res
    
    #random name
    letters = string.ascii_lowercase
    random_filename = f"output_{''.join(random.choices(letters, k=10))}.jpg"
    
    output_folder = "OUTPUT"
    os.makedirs(output_folder, exist_ok=True)
    
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check for 3-channel image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    filepath = os.path.join(output_folder, random_filename)
    cv.imwrite(filepath, image)
    
def detection(image):
    global res
    
    i = 0
    model = YOLO(model_file)        
    
    result = model(image)
    
    # Save and update output for each detection
    for r in result:
        flname = f'./result/result_{i}.jpg'  
        r.save(flname) 

        detected_image_path = os.path.join('./result', f"result_{i}.jpg")  # Access saved result 
    
        # Update output image display
        res = cv.imread(detected_image_path)  
        res = cv.cvtColor(res, cv.COLOR_BGR2RGB)  
    
        height, width = res.shape[:2]
        
        width_ratio = float(max_width) / width
        height_ratio = float(max_height) / height
        
        resize_factor = min(width_ratio, height_ratio)

        
        width_new = int(width * resize_factor)
        height_new = int(height * resize_factor)
        img_resized = cv.resize(res, (width_new, height_new), interpolation=cv.INTER_AREA)
    
        detected_img = ImageTk.PhotoImage(Image.fromarray(img_resized))
        detected_img_label.configure(image=detected_img)
        detected_img_label.image = detected_img  # Keep reference
    
        i += 1  
    
    return 0

def main ():
    
    #Main Menu Frame
    global input_frame
    input_frame_width = screen_width * 0.96
    input_frame_height = screen_height * 0.07
    
    
    input_frame = tk.ttk.Frame(window, height=input_frame_height, width= input_frame_width, borderwidth = 20, relief='groove')
    input_frame.place(x=40, y= 5)
    
    
    #input button
    input_image = tk.Button(input_frame, text="Insert Image", width = 25, font =('times', 16, 'bold'), command = lambda : open_image())
    input_image.place(relx = 0.2, rely=0.5,anchor = 'center', width=250)
    
    
    #detect button
    detect = tk.Button(input_frame, text="DETECT IMAGE", width = 25, font =('times', 16, 'bold'),bd =5, fg = 'white',bg = 'blue', relief = 'raised', command = lambda : detection(filename))
    detect.place(relx = 0.5, rely=0.5,anchor = 'center', width=250)
    
    
    #save output image
    input_image = tk.Button(input_frame, text="Save Output Image", width = 25, font =('times', 16, 'bold'), command = lambda : save_image(res))
    input_image.place(relx = 0.8, rely=0.5,anchor = 'center', width=250)
    
    #Input Display
    global frame2
    input_display_width = screen_width * 0.47
    input_display_height = screen_height * 0.85
    
    frame2 = tk.ttk.Frame(window,height=input_display_height, width= input_display_width, borderwidth = 20, relief='groove')
    frame2.place(x=40, y= 100)
    
    l2 = tk.ttk.Label(frame2,text='Input Image',font=('times',18,'bold'))  
    l2.place(relx = 0.5, rely=0,anchor = 'n')
    
    global input_img_label
    input_img_label = ttk.Label(frame2)
    input_img_label.place(relx = 0.5, rely=0.5,anchor = 'center')
    
    #output image frame
    global frame1
    output_display_width = screen_width * 0.47
    output_display_height = screen_height * 0.85
    
    
    frame1 = tk.ttk.Frame(window,height=output_display_height, width= output_display_width, borderwidth = 20, relief='groove')
    frame1.place(x=980, y= 100)
    
    l1 = tk.ttk.Label(frame1,text='Output Image',font=('times',18,'bold'))  
    l1.place(relx = 0.5, rely=0,anchor = 'n')
    
    global detected_img_label
    detected_img_label = ttk.Label(frame1)
    detected_img_label.place(relx = 0.5, rely=0.5,anchor = 'center')
    
    window.mainloop()
    

if __name__ == "__main__":
    main()