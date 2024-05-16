from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte

from PIL import Image
from PIL.ExifTags import TAGS
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Frame, Image as Images
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus.flowables import KeepInFrame
from stegano import lsb
from skimage import io, img_as_ubyte, img_as_float
from skimage.restoration import estimate_sigma
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import io as iolib
import imageio
import numpy as np
from skimage import filters, color, feature
from io import BytesIO
import cv2
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use('Agg')

import subprocess
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import string




import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER







#VIDEO FORENSICS
#METADATA ANALYSIS
def extract_metadata(video_path):
    try:
        # Run ExifTool with the specified file path
        result = subprocess.run(['exiftool', video_path], capture_output=True, text=True, check=True)

        # Extract and return the output
        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        # Handle any errors
        print("Error:", e)
        return None


#COMPRESSION ARTIFACT ANALYSIS
def spatial_artifact_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply blockiness detection
    blockiness_map = cv2.dct(np.float32(gray_image))
    blockiness_score = np.mean(blockiness_map)
    
    # Apply blur detection
    blur_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    # Apply noise analysis
    noise_score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    
    # Apply edge analysis
    edges = cv2.Canny(gray_image, 100, 200)
    edge_score = np.mean(edges)
    
    return blockiness_score, blur_score, noise_score, edge_score

def temporal_artifact_detection(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize variables for temporal analysis
    frame_count = 0
    motion_scores = []
    frame_rate_changes = 0
    selected_frames = []
    
    # Iterate through video frames
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1
        
        # Apply motion analysis
        if frame_count > 1:
            prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_score = np.mean(np.abs(flow))
            motion_scores.append(motion_score)
            
            # Select frames with significant temporal artifacts
            if motion_score > 0.5:
                selected_frames.append((frame_count, frame))
        
        # Apply frame rate analysis
        if frame_count > 1:
            prev_frame_rate = cap.get(cv2.CAP_PROP_FPS)
            current_frame_rate = cap.get(cv2.CAP_PROP_FPS)
            if prev_frame_rate != current_frame_rate:
                frame_rate_changes += 1
        
        # Store current frame as previous frame
        prev_frame = frame.copy()
    
    # Calculate average motion score and frame rate changes
    avg_motion_score = np.mean(motion_scores)
    avg_frame_rate_changes = frame_rate_changes / frame_count
    
    # Release the video capture object
    cap.release()
    
    return avg_motion_score, avg_frame_rate_changes, selected_frames


def spatial_artifact_detection_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Initialize lists to store spatial artifact scores for each frame
    blockiness_scores = []
    blur_scores = []
    noise_scores = []
    edge_scores = []
    selected_frames = []
    
    # Iterate through video frames
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform spatial artifact detection for the current frame
        blockiness_score, blur_score, noise_score, edge_score = spatial_artifact_detection(frame)
        
        # Store the scores for the current frame
        blockiness_scores.append(blockiness_score)
        blur_scores.append(blur_score)
        noise_scores.append(noise_score)
        edge_scores.append(edge_score)
        
        # Select frames with significant spatial artifacts
        if blockiness_score > 0.5:
            selected_frames.append((cap.get(cv2.CAP_PROP_POS_FRAMES), frame))
    
    # Release the video capture object
    cap.release()
    
    return blockiness_scores, blur_scores, noise_scores, edge_scores, selected_frames

def compression_artifact(video_path):
    blockiness_scores, blur_scores, noise_scores, edge_scores, selected_spatial_frames = spatial_artifact_detection_video(video_path)

    # Plot spatial artifact analysis results for the video
    frames = np.arange(len(blockiness_scores))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(frames, blockiness_scores, color='blue')
    plt.title('Blockiness Score Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Blockiness Score')

    plt.subplot(2, 2, 2)
    plt.plot(frames, blur_scores, color='green')
    plt.title('Blur Score Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Blur Score')

    plt.subplot(2, 2, 3)
    plt.plot(frames, noise_scores, color='orange')
    plt.title('Noise Score Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Noise Score')

    plt.subplot(2, 2, 4)
    plt.plot(frames, edge_scores, color='red')
    plt.title('Edge Score Over Frames')
    plt.xlabel('Frame Number')
    plt.ylabel('Edge Score')

    plt.tight_layout()
    
    comp_stat_img_byteio = BytesIO()
    plt.savefig(comp_stat_img_byteio, format='png', bbox_inches='tight')
    plt.close()

    # Perform temporal artifact detection on the video
    avg_motion_score, avg_frame_rate_changes, selected_temporal_frames = temporal_artifact_detection(video_path)

    # Display frames with significant spatial artifacts
    plt.figure(figsize=(12, 5))
    plt.suptitle('Frames with Significant Spatial Artifacts', fontsize=16)

    for idx, (frame_number, frame) in enumerate(selected_spatial_frames[:9]):
        plt.subplot(3, 3, idx + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Frame {}'.format(int(frame_number)))
        plt.axis('off')

    plt.tight_layout()
    spatial_image_byteio = BytesIO()
    plt.savefig(spatial_image_byteio, format='png', bbox_inches='tight')
    plt.close()

    # Display frames with significant temporal artifacts
    plt.figure(figsize=(12, 5))
    plt.suptitle('Frames with Significant Temporal Artifacts', fontsize=16)

    for idx, (frame_number, frame) in enumerate(selected_temporal_frames[:9]):
        plt.subplot(3, 3, idx + 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title('Frame {}'.format(int(frame_number)))
        plt.axis('off')

    plt.tight_layout()
    temporal_image_byteio = BytesIO()
    plt.savefig(temporal_image_byteio, format='png', bbox_inches='tight')
    plt.close()
    
    return comp_stat_img_byteio.getvalue(), spatial_image_byteio.getvalue(), temporal_image_byteio.getvalue()


#NOISE ANALYSIS
def noise_analysis_video(video_path, window_size=5):
    top_n = 9
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize arrays to store pixel values and noise estimates
    pixel_values = np.zeros((frame_count, height, width))
    noise_estimates = np.zeros((frame_count,))
    
    # Read video frames and store pixel values
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        pixel_values[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate noise estimates using variance of pixel values across frames
    for i in range(frame_count):
        start_frame = max(0, i - window_size // 2)
        end_frame = min(frame_count, i + window_size // 2 + 1)
        pixel_window = pixel_values[start_frame:end_frame]
        noise_estimates[i] = np.var(pixel_window)
    
    
    sorted_frames = np.argsort(noise_estimates)[::-1][:top_n]
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display top frames
    for idx, frame_idx in enumerate(sorted_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            plt.subplot(3, 3, idx + 1)
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('Frame {}'.format(frame_idx))
            plt.axis('off')
            
            # Add variance value and frame number below the plot
            plt.text(0.5, -0.1, 'Variance {:.2f}'.format(noise_estimates[frame_idx]),
                     horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    # Release video capture
    cap.release()
    
    plt.tight_layout()
    noise_img_bytesio = BytesIO()
    plt.savefig(noise_img_bytesio, format='png', bbox_inches='tight')
    plt.close()
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(noise_estimates, color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Noise Level')
    plt.grid(True)
    
    noise_histo_bytesio = BytesIO()
    plt.savefig(noise_histo_bytesio, format='png', bbox_inches='tight')
    plt.close()
    
    return noise_histo_bytesio.getvalue(), noise_img_bytesio.getvalue()
    
    



def generate_random_string(size=7):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))



def get_downloads_folder():
    """ Returns the path of the downloads folder for different operating systems. """
    if os.name == 'nt':  # For Windows
        return os.path.join(os.environ['USERPROFILE'], 'Downloads')
    else:  # For macOS and Linux
        return os.path.join(os.path.expanduser('~'), 'Downloads')



# Creating Model Architecture
class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    path_to_videos= [videoPath]

    video_dataset = validation_dataset(path_to_videos,sequence_length = 20,transform = train_transforms)
    # use this command for gpu
    # model = Model(2).cuda()
    model = Model(2)
    path_to_model = 'model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model,video_dataset[i],'./')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction



def generate_forensic_report_video(video_path):
    downloads_folder = get_downloads_folder()  # Get the path to the downloads folder
    video_filename = video_path.split("/")[-1]  
    filename_prefix = video_filename.split(".")[0]
    random_string = generate_random_string()
    report_filename = os.path.join(downloads_folder, f"{filename_prefix}_{random_string}_forensic-report.pdf")
    doc = SimpleDocTemplate(report_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Heading style
    heading_style = ParagraphStyle(
        name="Heading1",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.black,
        spaceAfter=10
    )
    center_style = ParagraphStyle(
        name="center", alignment=1, fontSize=20, textColor=colors.blue
    )

    normal_style = ParagraphStyle(
    name="Normal",
    fontName="Helvetica",
    fontSize=12,  # Adjust font size as needed
    textColor=colors.black,
    alignment=0
    )

    original_label_style = ParagraphStyle(
        name="OriginalImageLabel",
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.black,
        alignment=1  # Center alignment
    )


    content = []

    heading = Paragraph("VIDEO FORENSIC REPORT", center_style)
    content.append(heading)
    content.append(Spacer(1, 20))
    content.append(Spacer(1, 40))


    metadata_output = extract_metadata(video_path)
    metadata_heading = Paragraph("Metadata Analysis:", heading_style)
    metadata_paragraph = Paragraph(metadata_output, styles["Normal"])
    content.append(metadata_heading)
    for line in metadata_output.split('\n'):
        content.append(Paragraph(line, styles["Normal"]))

    content.append(Spacer(1, 20))
    
    
    
    
    compstat_bytes, spatial_bytes, temporal_bytes = compression_artifact(video_path)
    compstat_image = ReportLabImage(BytesIO(compstat_bytes))
    spatial_image = ReportLabImage(BytesIO(spatial_bytes))
    temporal_image = ReportLabImage(BytesIO(temporal_bytes))
    compstat_image.drawWidth = 500
    compstat_image.drawHeight = 300
    compstat_image.hAlign = "CENTER"
    spatial_image.drawWidth = 400
    spatial_image.drawHeight = 200
    spatial_image.hAlign = "CENTER"
    temporal_image.drawWidth = 400
    temporal_image.drawHeight = 200
    temporal_image.hAlign = "CENTER"
    content.append(Paragraph("Compression Artifact Analysis:", heading_style))
    content.append(Spacer(1, 20))
    content.append(compstat_image)
    content.append(Spacer(1, 20))
    content.append(Paragraph("The plot above provides an analysis of various spatial artifacts present in the video frames. This analysis focuses on four key spatial artifact metrics: Blockiness, Blur, Noise, and Edge.", normal_style))
    content.append(Spacer(1, 10))
    sp1 = "1. Blockiness Score: The blue line in the top-left subplot represents the blockiness score computed for each frame of the video. Higher blockiness scores indicate a higher level of blockiness artifacts in the video frames."
    sp2 = "2. Blur Score: The green line in the top-right subplot represents the blur score calculated for each frame. Blur detection measures the level of blurriness present in the video frames, which can result from various factors such as camera motion, focus issues, or compression artifacts. Higher blur scores indicate greater blurriness in the frames."
    sp3 = "3. Noise Score: The orange line in the bottom-left subplot illustrates the noise score across the frames. Higher noise scores suggest a higher level of noise interference in the frames."
    sp4 = "4. Edge Score: The red line in the bottom-right subplot displays the edge score over the frames. Edge detection identifies abrupt changes in pixel intensity, often corresponding to object boundaries or edges within the video frames. Higher edge scores indicate a greater abundance of edges detected in the frames."
    content.append(Paragraph(sp1, normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph(sp2, normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph(sp3, normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph(sp4, normal_style))
    content.append(Spacer(1, 40))
    content.append(spatial_image)
    content.append(Spacer(1, 20))
    content.append(Paragraph("The grid above displays frames from the video where significant spatial artifacts were detected. These frames highlight areas where distortions or irregularities are prominent, indicating potential video quality issues.", normal_style))
    content.append(Spacer(1, 40))
    content.append(temporal_image)
    content.append(Spacer(1, 20))
    content.append(Paragraph("The grid above presents frames from the video where significant temporal artifacts (motion) were detected. These frames highlight instances where notable motion or changes over time were observed, indicating potential video quality issues.", normal_style))
    content.append(Spacer(1, 40))
    
    
    
    noise_histo_bytes, noise_img_bytes = noise_analysis_video(video_path, window_size=5)
    noise_histo = ReportLabImage(BytesIO(noise_histo_bytes))
    noise_img = ReportLabImage(BytesIO(noise_img_bytes))
    noise_histo.drawWidth = 300
    noise_histo.drawHeight = 300
    noise_histo.hAlign = 'CENTER'
    noise_img.drawWidth = 400
    noise_img.drawHeight = 250
    noise_img.hAlign = 'CENTER'
    content.append(Paragraph("Noise Analysis:", heading_style))
    content.append(Spacer(1, 10))
    content.append(noise_histo)
    content.append(Spacer(1, 20))
    content.append(Paragraph("The plot above displays the noise level analysis results for each frame of the video. The x-axis represents the frame number, and the y-axis represents the noise level. A higher noise level indicates more significant noise present in the video frames.", normal_style))
    content.append(Spacer(1, 50))
    content.append(noise_img)
    content.append(Spacer(1, 20))
    content.append(Paragraph("The grid of frames displayed above highlights the frames with the most pronounced noise levels identified during the noise analysis process. Each frame within the grid is annotated with its corresponding frame number, facilitating easy reference to the video timeline. Additionally, beneath each frame, the variance value, representing the degree of noise present within the frame, is provided.", normal_style))
    content.append(Spacer(1, 40))
    

    prediction = detectFakeVideo(video_path)
    predict = prediction[0]
    confidence = prediction[1] 
    if predict == 0:
        res = "FAKE"
    else:
        res = "REAL"
    content.append(Paragraph("Analysis using Deep Learning", heading_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph("Detection of video using Deep Learning(ResNext and LSTM):", normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph("Result: "+ str(res), normal_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Confidence: "+ str(confidence), normal_style))
    content.append(Spacer(1, 20))
    


    doc.build(content)
    print(f"Forensic report '{report_filename}' generated successfully and saved to Downloads folder.")




    
#IMAGE FORENSICS
# Function to extract metadata from the image
def extract_metadata(image_path):
    try:
        # Run ExifTool with the specified file path
        result = subprocess.run(['exiftool', image_path], capture_output=True, text=True, check=True)

        # Extract and return the output
        output = result.stdout.strip()
        return output
    except subprocess.CalledProcessError as e:
        # Handle any errors
        print("Error:", e)
        return None



# Function to detect steganography in the image
def detect_steganography(image_path):
    steganography_info = ""
    try:
        image = Image.open(image_path)
        secret_message = lsb.reveal(image)

        if secret_message:
            steganography_info += "Steganography Detection:\n"
            steganography_info += secret_message
        else:
            steganography_info += "No steganography detected in the image."

    except Exception as e:
        steganography_info += f"Error: {e}"

    return steganography_info


def perform_ela(input_image_path, quality=75, scale=15):
    # Read the input image
    input_image = io.imread(input_image_path)

        # Apply Gaussian filter and convert to appropriate format
    filtered_image = gaussian(input_image, sigma=estimate_sigma(input_image))
    compressed_image = img_as_ubyte(filtered_image)

        # Define the directory path where you want to save the temporary file
    directory_path = 'Uploaded_Files'

        # Ensure the directory exists, create if it doesn't
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

        # Construct the full path to save the temporary image
    temp_file_path = os.path.join(directory_path, 'temp.jpg')

        # Save the compressed image to the specified directory
    io.imsave(temp_file_path, compressed_image)

        # Read the compressed image from the temporary file
    compressed_image = io.imread(temp_file_path)

    # Calculate the ELA
    ela = img_as_float(input_image) - img_as_float(compressed_image)

    # Multiply by the scale factor
    ela *= scale

    # Create a BytesIO object to store the ELA image
    ela_bytes_io = iolib.BytesIO()

    # Save the ELA image to the BytesIO object
    plt.imshow(ela)
    plt.axis("off")
    plt.savefig(ela_bytes_io, format='png', bbox_inches='tight')
    plt.close()

    return ela_bytes_io.getvalue()


def forgery_detection(image_path):
    try:
        # Read the image
        image = imageio.imread(image_path)

        # Convert the image to grayscale
        grayscale_image = color.rgb2gray(image)

        # Apply edge detection using the Canny filter
        edges = feature.canny(grayscale_image, sigma=3)

        # Calculate the gradient magnitude
        gradient_magnitude = filters.sobel(grayscale_image)

        # Create a figure with subplots for displaying images
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Display the original image
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # Display the edges detected using the Canny filter
        ax[1].imshow(edges, cmap=plt.cm.gray)
        ax[1].set_title("Edges (Canny Filter)")
        ax[1].axis("off")

        # Display the gradient magnitude
        ax[2].imshow(gradient_magnitude, cmap=plt.cm.gray)
        ax[2].set_title("Gradient Magnitude")
        ax[2].axis("off")

        # Adjust layout to prevent overlap of titles
        plt.tight_layout(pad=1.0)

        # Save the figure to a BytesIO object
        forgery_image_bytes_io = BytesIO()
        plt.savefig(forgery_image_bytes_io, format='png', bbox_inches='tight')
        plt.close()

        return forgery_image_bytes_io.getvalue()

    except Exception as e:
        print(f"Error: {e}")


def echo_edge_filter(image_path, num_iterations=5, edge_threshold_low=50, edge_threshold_high=150):
    # Load the input image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load the image in color mode

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to obtain initial edge map
    edge_map = cv2.Canny(gray, edge_threshold_low, edge_threshold_high)

    # Apply the echo effect
    for _ in range(num_iterations):
        edge_map = cv2.Canny(edge_map, edge_threshold_low, edge_threshold_high)

    # Convert the edge map to color for visualization
    edge_map_color = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    # Combine the edge map with the original image
    output_image = cv2.addWeighted(img, 0.5, edge_map_color, 0.5, 0)

    # Create a plot and save it to a byte stream
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Echo Edge Filtered Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edge_map, cmap='gray')
    plt.title("Final Edge Map")
    plt.axis('off')

    plt.tight_layout(pad=2.0)

    # Save the plot to a byte stream
    echo_image_bytes_io = BytesIO()
    plt.savefig(echo_image_bytes_io, format='png', bbox_inches='tight')
    plt.close()  # Corrected to properly close the matplotlib plot

    return echo_image_bytes_io.getvalue()




def compression_artifact_analysis(image_path, quality):
    try:
        # Read the original image
        original_image = cv2.imread(image_path)

        # Define the directory path where you want to save the original image
        directory_path = 'Uploaded_Files'

        # Ensure the directory exists, create if it doesn't
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Construct the full path to save the original image
        original_file_path = os.path.join(directory_path, 'original.jpg')

        # Save the original image with the specified quality to the specified directory
        cv2.imwrite(original_file_path, original_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # Read the re-compressed image from the saved file
        recompressed_image = cv2.imread(original_file_path)

        # Calculate the absolute difference between original and re-compressed images
        diff_image = cv2.absdiff(original_image, recompressed_image)

        # Convert the difference image to grayscale
        diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

        # Threshold the difference image to highlight artifacts
        _, thresholded_diff = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        # Calculate the percentage of pixels affected by compression artifacts
        total_pixels = original_image.shape[0] * original_image.shape[1]
        affected_pixels = np.count_nonzero(thresholded_diff)
        artifact_percentage = (affected_pixels / total_pixels) * 100

        # Create a figure with subplots for displaying images
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        # Display the original image
        ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        # Display the recompressed image
        ax[1].imshow(cv2.cvtColor(recompressed_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Recompressed Image")
        ax[1].axis("off")

        # Display the difference image
        ax[2].imshow(thresholded_diff, cmap='gray')
        ax[2].set_title('Difference Image (Artifact Highlighted)')
        ax[2].axis('off')

        # Adjust layout to prevent overlap of titles
        plt.tight_layout(pad=1.0)

        # Save the figure to a BytesIO object
        compression_artifact_image_bytes_io = BytesIO()
        plt.savefig(compression_artifact_image_bytes_io, format='png', bbox_inches='tight')
        plt.close()

        return compression_artifact_image_bytes_io.getvalue(), artifact_percentage

    except Exception as e:
        print(f"Error: {e}")



def color_analysis(image_path):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Flatten the image to a 2D array of RGB pixels
    pixels = image.reshape((-1, 3))

    # Calculate mean color
    mean_color = np.mean(pixels, axis=0)

    # Calculate dominant colors
    color_counts = Counter(map(tuple, pixels))
    dominant_colors = [list(color) for color, _ in color_counts.most_common(5)]

    # Plot histogram of color distribution
    color_distribution_fig = BytesIO()
    plt.figure(figsize=(10, 6))
    plt.hist2d(pixels[:, 0], pixels[:, 1], bins=(32, 32), cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('Color Distribution')
    plt.xlabel('Red')
    plt.ylabel('Green')
    plt.savefig(color_distribution_fig, format='png', bbox_inches='tight')
    plt.close()

    # Display the image with dominant colors highlighted
    dominant_colors_fig = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title('Image with Dominant Colors Highlighted')
    for color in dominant_colors:
        plt.plot([0], [0], marker='o', markersize=10, color=np.array(color)/255.0, label=str(color))
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.savefig(dominant_colors_fig, format='png', bbox_inches='tight')
    plt.close()

    # Print results
    mean_color_text = f"Mean Color (RGB): {mean_color}"
    dominant_colors_text = "Dominant Colors (RGB):"
    for color in dominant_colors:
        dominant_colors_text += f"\n{color}"

    return mean_color_text, dominant_colors_text, color_distribution_fig, dominant_colors_fig



def pixel_statistics(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Split channels
    b, g, r = cv2.split(image)

    # Calculate histograms for each channel
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    # Calculate statistics
    mean_intensity_r = np.mean(r)
    mean_intensity_g = np.mean(g)
    mean_intensity_b = np.mean(b)

    # Compute minimum, maximum, and average RGB values for every pixel
    min_rgb_values = np.min(image, axis=(0, 1))
    max_rgb_values = np.max(image, axis=(0, 1))
    avg_rgb_values = np.mean(image, axis=(0, 1))


    # Create an image with RGB colors based on statistics
    new_image = np.zeros_like(image)
    new_image[:,:,0] = int(mean_intensity_b)  # Blue channel
    new_image[:,:,1] = int(mean_intensity_g)  # Green channel
    new_image[:,:,2] = int(mean_intensity_r)  # Red channel

    # Display the histogram
    plt.figure(figsize=(12, 8))
    plt.plot(hist_r, color='red', label='Red')
    plt.plot(hist_g, color='green', label='Green')
    plt.plot(hist_b, color='blue', label='Blue')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histograms of Pixel Intensities (RGB)')
    plt.legend()
    plt.grid(True)
    
    pixel_statimg_bytesio = BytesIO()
    plt.savefig(pixel_statimg_bytesio, format='png', bbox_inches='tight')
    plt.close()

    return min_rgb_values, max_rgb_values, avg_rgb_values, pixel_statimg_bytesio.getvalue()




def composite_splicing(image_path):
    # Load the input image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute the Sobel gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Normalize the gradient magnitude to the range [0, 255]
    gradient_magnitude_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create a probability heatmap
    probability_heatmap = cv2.applyColorMap(gradient_magnitude_norm, cv2.COLORMAP_JET)

    plt.figure(figsize=(8, 6))
    plt.imshow(probability_heatmap)
    plt.colorbar()
    plt.title("Splicing Probability Heatmap")
    plt.axis('off')

    heatmap_image_bytes_io = BytesIO()
    plt.savefig(heatmap_image_bytes_io, format='png', bbox_inches='tight')
    plt.close()

    return heatmap_image_bytes_io.getvalue()
    



def texture_analysis(image_path):
    try:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Calculate GLCM properties
        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')

        # Plotting the texture analysis results
        properties = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
        results = [contrast, dissimilarity, homogeneity, energy, correlation]

        images = []

        # First three images in a single horizontal frame
        plt.figure(figsize=(15, 5))
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(results[i], cmap='hot', interpolation='nearest')
            plt.title(properties[i])
            plt.colorbar()
            plt.axis('off')
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        plt.close()
        img_io.seek(0)
        images.append(img_io)

        # Next two images in another horizontal frame
        plt.figure(figsize=(10, 5))
        for i in range(3, 5):
            plt.subplot(1, 2, i-2)
            plt.imshow(results[i], cmap='hot', interpolation='nearest')
            plt.title(properties[i])
            plt.colorbar()
            plt.axis('off')
        img_io = BytesIO()
        plt.savefig(img_io, format='png')
        plt.close()
        img_io.seek(0)
        images.append(img_io)

        return images, properties, contrast, dissimilarity, homogeneity, energy, correlation

    except Exception as e:
        print(f"Error: {e}")
        return [], [], None, None, None, None, None


def resampling_detection(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter to detect edges
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)

    # Threshold the Sobel image
    _, thresholded = cv2.threshold(sobel, 10, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels
    nonzero_pixels = cv2.countNonZero(thresholded)

    # Calculate percentage of non-zero pixels
    total_pixels = gray.shape[0] * gray.shape[1]
    percent_nonzero = (nonzero_pixels / total_pixels) * 100

    # Plot original image and edges
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(thresholded, cmap='gray')
    axes[1].set_title('Detected Edges')
    axes[1].axis('off')

    resamp_image_bytes_io = BytesIO()
    plt.savefig(resamp_image_bytes_io, format='png', bbox_inches='tight')
    plt.close()

    return percent_nonzero, resamp_image_bytes_io.getvalue()




def noise_analysis(image_path):
    try:
        # Read the input image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the noise in the image
        noise = gray_image - cv2.GaussianBlur(gray_image, (0, 0), 1)

        # Save the images to a single horizontal frame
        plt.figure(figsize=(15, 5))
        plt.subplots_adjust(wspace=0.1)
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image (Color)')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Original Image (Grayscale)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(noise, cmap='gray')
        plt.title('Noise Image')
        plt.axis('off')
        horizontal_frame_io = BytesIO()
        plt.savefig(horizontal_frame_io, format='png')
        horizontal_frame_io.seek(0)
        plt.close()

        # Plot pixel intensity histogram
        plt.hist(gray_image.ravel(), 256, [0, 256], color='blue', alpha=0.5, label='Original Image (Grayscale)')
        plt.hist(noise.ravel(), 256, [0, 256], color='red', alpha=0.5, label='Noise')
        plt.title('Pixel Intensity Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt_histogram = BytesIO()
        plt.savefig(plt_histogram, format='png')
        plt_histogram.seek(0)
        plt.close()

        # Calculate the mean and standard deviation of the noise
        mean_noise = np.mean(noise)
        std_dev_noise = np.std(noise)

        return horizontal_frame_io, plt_histogram, mean_noise, std_dev_noise

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None




# Function to generate the forensic report PDF
def generate_forensic_report_image(image_path):
    downloads_folder = get_downloads_folder()
    image_filename = image_path.split("/")[-1]  
    filename_prefix = image_filename.split(".")[0]
    random_string = generate_random_string(size=7)
    report_filename = os.path.join(downloads_folder, f"{filename_prefix}_{random_string}_forensic-report.pdf")
    doc = SimpleDocTemplate(report_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Heading style
    heading_style = ParagraphStyle(
        name="Heading1",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.black,
        spaceAfter=10
    )
    center_style = ParagraphStyle(
        name="center", alignment=1, fontSize=20, textColor=colors.blue
    )

    normal_style = ParagraphStyle(
    name="Normal",
    fontName="Helvetica",
    fontSize=12,  # Adjust font size as needed
    textColor=colors.black,
    alignment=0
    )

    original_label_style = ParagraphStyle(
        name="OriginalImageLabel",
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.black,
        alignment=1  # Center alignment
    )


    content = []

    heading = Paragraph("IMAGE FORENSIC REPORT", center_style)
    content.append(heading)
    content.append(Spacer(1, 20))
    content.append(Spacer(1, 40))

    centered_img = ReportLabImage(image_path, width=250, height=250)
    centered_img.hAlign = "CENTER"
    content.append(centered_img)

    original_label = Paragraph("UPLOADED IMAGE", original_label_style)
    content.append(original_label)
    content.append(Spacer(1, 30))

    metadata_output = extract_metadata(image_path)
    metadata_heading = Paragraph("Metadata Analysis:", heading_style)
    metadata_paragraph = Paragraph(metadata_output, styles["Normal"])
    content.append(metadata_heading)
    for line in metadata_output.split('\n'):
        content.append(Paragraph(line, styles["Normal"]))

    content.append(Spacer(1, 20))

    

    steganography_output = detect_steganography(image_path)
    steganography_heading = Paragraph("Steganography Detection:", heading_style)
    steganography_paragraph = Paragraph(steganography_output, styles["Normal"])
    content.append(steganography_heading)
    content.append(steganography_paragraph)
    content.append(Spacer(1, 20))


    ela_bytes_io = perform_ela(image_path)
    ela_image = ReportLabImage(iolib.BytesIO(ela_bytes_io))
    ela_image.drawWidth = 250
    ela_image.drawHeight = 250
    ela_image.hAlign = "CENTER"
    content.append(Paragraph("Error Level Analysis:", heading_style))
    content.append(ela_image)
    content.append(Spacer(1, 20))

    ela_description = """
    The ELA image highlights areas of the image where alterations may have occurred compared to the original image. These alterations typically manifest as regions with higher contrast or intensity, indicating potential modifications. Darker or more pronounced areas within the ELA image are suggestive of alterations, while lighter or less prominent regions are likely unchanged."""
    content.append(Paragraph(ela_description, normal_style))
    content.append(Spacer(1, 20))


    forgery_image_bytes = forgery_detection(image_path)
    forgery_image = ReportLabImage(BytesIO(forgery_image_bytes))
    forgery_image.drawWidth = 500  # Adjust width as needed
    forgery_image.drawHeight = 200    
    forgery_image.hAlign = "CENTER"
    forgery_detection_heading = Paragraph("Forgery Detection:", heading_style)
    content.append(forgery_detection_heading)
    content.append(forgery_image)
    content.append(Spacer(1, 10))
    lines = ["This is the original image without any processing.",
         "Edges detected using the Canny filter. These edges represent abrupt changes in intensity in the image.",
         "Gradient magnitude analysis highlights regions with significant intensity changes, which may indicate potential areas of manipulation or tampering."]
    headings = ["Original Image:", "Edges (Canny Filter):", "Gradient Magnitude:"]
    for heading, line in zip(headings, lines):
        content.append(Paragraph(f"{heading} {line}", normal_style))
        content.append(Spacer(1, 5))  
    content.append(Spacer(1, 20))


    echo_image_bytes = echo_edge_filter(image_path, num_iterations=5, edge_threshold_low=50, edge_threshold_high=150)
    echo_image = ReportLabImage(BytesIO(echo_image_bytes))
    echo_image.drawWidth = 400
    echo_image.drawHeight = 200
    echo_image.hAlign = "CENTER"
    echo_heading = Paragraph("Echo Edge Filter: ", heading_style)
    content.append(echo_heading)
    content.append(Spacer(1, 10))
    content.append(echo_image)
    content.append(Spacer(1, 10))
    content.append(Paragraph("Echo Edge Filtered Image: The echo edge filtered image highlights and emphasizes the edges in the original image, creating a layered 'echo' effect.", normal_style))
    content.append(Spacer(1, 20))


    image_quality = 50  # Adjust the JPEG compression quality (0-100)
    compression_artifact_image_bytes, artifact_percentage = compression_artifact_analysis(image_path, image_quality)
    compression_artifact_explanation = (
    "Compression artifacts are distortions in an image caused by compression, notably in formats like JPEG. "
    "The artifact percentage quantifies the extent of these distortions:"
    )
    e1 = "1. If the artifact percentage is below 1%, it suggests that there are negligible compression artifacts detected. This indicates that the image quality remains high, with minimal impact from compression."
    e2 = "2. When the artifact percentage is between 1% and 5%, some compression artifacts may be present, but the overall image quality is generally preserved. These artifacts might be subtle and may not significantly affect visual perception."
    e3 = "3. If the artifact percentage exceeds 5%, it indicates the presence of significant compression artifacts, which can noticeably impact image quality. These artifacts may manifest as blockiness, blurring, or color inaccuracies, detracting from the visual fidelity of the image."
    compression_artifact_image = ReportLabImage(BytesIO(compression_artifact_image_bytes))
    compression_artifact_image.drawWidth = 500  # Adjust width as needed
    compression_artifact_image.drawHeight = 200
    compression_artifact_image.hAlign = "CENTER"
    content.append(Paragraph("Compression Artifact Analysis:", heading_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(compression_artifact_explanation, normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph(e1, normal_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(e2, normal_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(e3, normal_style))
    content.append(Spacer(1, 20))
    content.append(compression_artifact_image)
    content.append(Spacer(1, 10))
    explanation = ""
    if artifact_percentage < 1:
        explanation = "Negligible compression artifacts detected. Image quality remains high."
    elif 1 <= artifact_percentage < 5:
        explanation = "Some compression artifacts detected, but image quality is generally preserved."
    else:
        explanation = "Significant compression artifacts detected, impacting image quality."
    content.append(Paragraph(f"Compression Artifact Percentage: {artifact_percentage:.2f}%", normal_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(explanation, normal_style))
    content.append(Spacer(1, 20))



    mean_color_text, dominant_colors_text, color_distribution_fig, dominant_colors_fig = color_analysis(image_path)
    color_distribution_image = ReportLabImage(color_distribution_fig, width=400, height=250)
    color_distribution_image.hAlign = "CENTER"
    dominant_colors_highlighted_image = ReportLabImage(dominant_colors_fig, width=250, height=250)
    dominant_colors_highlighted_image.hAlign = "CENTER"
    content.append(Paragraph("Color Analysis:", heading_style))
    content.append(color_distribution_image)
    content.append(Spacer(1, 10))
    content.append(Paragraph("The histogram above represents the distribution of colors in the image. Each bin in the histogram corresponds to a range of color intensities. The X-axis represents the intensity of the red channel, while the Y-axis represents the intensity of the green channel. The color density in each bin indicates the frequency of occurrence of colors within that intensity range.", normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph("A well-distributed histogram with a wide range of intensities indicates a diverse color palette in the image, while a histogram skewed towards a particular range suggests dominance of certain colors. Analyzing the distribution helps in understanding the overall color composition, identifying predominant colors, and detecting any color biases or anomalies in the image.", normal_style))
    content.append(Spacer(1, 10))
    content.append(dominant_colors_highlighted_image)
    content.append(Spacer(1, 10))
    content.append(Paragraph("The image above highlights the dominant colors present in the image. Each marker represents one of the top five most frequent colors found in the image, with its position corresponding to its RGB value. The presence of dominant colors provides insights into the overall color scheme and theme of the image.", normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph("By visually identifying and analyzing the dominant colors, one can gain a deeper understanding of the visual content, including identifying key elements, themes, or patterns. This analysis aids in various applications such as image classification, content-based image retrieval, and aesthetic evaluation.", normal_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Color Analysis Results:", normal_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph(mean_color_text, normal_style))
    content.append(Paragraph(dominant_colors_text, normal_style))
    content.append(Spacer(1, 20))


    min_col, max_col, mean_col, pixel_stat_bytes = pixel_statistics(image_path)
    pixel_statimg = ReportLabImage(BytesIO(pixel_stat_bytes))
    pixel_statimg.drawWidth = 250
    pixel_statimg.drawHeight = 250
    pixel_statimg.hAlign = "CENTER"
    content.append(Paragraph("Pixel Statistics:", heading_style))
    content.append(Spacer(1, 10))
    content.append(Paragraph("Minimum RGB values for every pixel: " + str(min_col), normal_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph("Maximum RGB values for every pixel: " + str(max_col), normal_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph("Average RGB values for every pixel: " + str(mean_col), normal_style))
    content.append(Spacer(1, 10))
    content.append(pixel_statimg)
    content.append(Spacer(1, 20))


    composite_image_bytes = composite_splicing(image_path)
    composite_image = ReportLabImage(BytesIO(composite_image_bytes))
    composite_image.drawWidth = 250
    composite_image.drawHeight = 250
    composite_image.hAlign = 'CENTER'
    content.append(Paragraph("Composite Splicing:", heading_style))
    content.append(Spacer(1, 10))
    content.append(composite_image)
    content.append(Spacer(1, 10))
    content.append(Paragraph("The splicing probability heatmap indicates the likelihood of splicing in the input image. Brighter regions in the heatmap indicate higher probabilities of splicing, while darker regions indicate lower probabilities.", normal_style))
    content.append(Spacer(1, 40))
    



    images, properties, contrast_value, dissimilarity_value, homogeneity_value, energy_value, correlation_value = texture_analysis(image_path)
    if images:
        reportlab_images = [Images(img) for img in images]
        for img in reportlab_images:
            img.drawWidth = 400  
            img.drawHeight = 200
            img.hAlign = "CENTER"
        content.append(Paragraph("Texture Analysis Results:", heading_style))
        content.append(Spacer(1, 10))
        content.append(Paragraph(f"Contrast: {contrast_value}", normal_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(f"Dissimilarity: {dissimilarity_value}", normal_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(f"Homogeneity: {homogeneity_value}", normal_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(f"Energy: {energy_value}", normal_style))
        content.append(Spacer(1, 5))
        content.append(Paragraph(f"Correlation: {correlation_value}", normal_style))
        for img in reportlab_images:
            content.append(img)
        content.append(Spacer(1, 10))
    else:
        content.append(Paragraph("Error: Texture analysis could not be performed.", normal_style))
        content.append(Spacer(1, 30))



    percentage, resamp_image_bytes = resampling_detection(image_path)
    resamp_image = ReportLabImage(BytesIO(resamp_image_bytes))
    resamp_image.drawWidth = 400
    resamp_image.drawHeight = 200
    resamp_image.hAlign = 'CENTER'
    content.append(Paragraph("Resampling Analysis:", heading_style))
    content.append(resamp_image)
    content.append(Spacer(1, 10))
    if percentage > 1:
        content.append(Paragraph("Resampling detected. Resampling trace percentage: " + str(percentage), normal_style))
    else:
        content.append(Paragraph("No resampling detected. Resampling trace percentage: " + str(percentage), normal_style))

    content.append(Spacer(1, 20))




    horizontal_frame_io, plt_histogram, mean_noise, std_dev_noise = noise_analysis(image_path)
    explanation_original_grayscale = (
    "The grayscale version of the original image simplifies the visualization by representing "
    "each pixel's intensity without considering color information, aiding in detecting subtle patterns or anomalies."
    )
    explanation_noise = (
    "The noise image highlights discrepancies or irregularities present in the image, which may stem from "
    "various sources such as sensor noise, compression artifacts, or digital manipulation. Analyzing noise patterns "
    "can provide valuable insights into the image's integrity and potential tampering."
    )
    description_histogram = (
    "The histogram plot provides a visual representation of the distribution of pixel intensities within the image.\n"
    "- The blue histogram represents the pixel intensity distribution of the original grayscale image. "
    "Each bin corresponds to a range of pixel intensities, and the height of the bar indicates the frequency of pixels within that range.\n"
    "- The red histogram depicts the pixel intensity distribution of the noise present in the image. "
    "By comparing it with the original image's histogram, you can identify regions where noise is more pronounced, potentially indicating areas of tampering or alteration."
    )
    noise_image = ReportLabImage(horizontal_frame_io, width=600, height=200)
    histogram_image = ReportLabImage(plt_histogram, width=400, height=200)
    noise_image.hAlign = "CENTER"
    histogram_image.hAlign = "CENTER"
    content.append(Paragraph("Noise Analysis:", heading_style))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Mean noise: {mean_noise}", normal_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(f"Standard deviation of noise: {std_dev_noise}", normal_style))
    content.append(noise_image)
    content.append(Spacer(1, 20))
    content.append(Paragraph(explanation_original_grayscale, normal_style))
    content.append(Spacer(1, 5))
    content.append(Paragraph(explanation_noise, normal_style))
    content.append(Spacer(1, 20))
    content.append(histogram_image)
    content.append(Spacer(1, 5))
    content.append(Paragraph(description_histogram, normal_style))


    doc.build(content)


def clean_folder(folder_path):
    try:
        # Get a list of all files in the folder
        file_list = os.listdir(folder_path)

        # Iterate over each file in the folder
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)

            # Check if the path is a file (not a directory)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")

        print(f"All files removed from {folder_path}")

    except Exception as e:
        print(f"Error cleaning folder: {e}")




@app.route('/', methods=['POST', 'GET'])
def homepage():
  if request.method == 'GET':
	  return render_template('index.html')
  return render_template('index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)
        filename = secure_filename(file.filename)
        file_type = file.content_type
        valid_image_types = ['image/jpeg', 'image/png']
        valid_video_types = ['video/mp4']
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = "Uploaded_Files/" + filename

        if file_type in valid_image_types:
            
            
            generate_forensic_report_image(file_path)
            print("Forensic report generated successfully.")
            data = {'output': "Forensic report generated successfully!!"}
            data = json.dumps(data)
            #os.remove(file_path)
            clean_folder("Uploaded_Files")

            return render_template('index.html', data=data)

        elif file_type in valid_video_types:

            data = {'output': "Forensic report generated successfully!!"}
            data = json.dumps(data)
            generate_forensic_report_video(file_path)
            print(f"Forensic report generated successfully.")
            clean_folder("Uploaded_Files") 
            #os.remove(file_path)
            

            return render_template('index.html', data=data)

        else:
            return jsonify({'error': 'Unsupported file type'}), 400
            
        
        
if __name__ == "__main__":
    app.run(port=3000, debug=True)