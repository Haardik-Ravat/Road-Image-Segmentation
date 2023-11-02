import streamlit as st
import torch

import torchvision.transforms as transforms

from PIL import Image
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.profiler
import torchvision.transforms as transforms

from matplotlib.colors import LinearSegmentedColormap

# Define your custom colors
colors = {
    0: [42.31849768, 138.68802268, 107.34976084],
    1: [127.04043101, 64.08636608, 127.54335584],
    2: [67.73835184, 70.81071668, 69.68390725],
    3: [77.50990865, 18.95552339, 73.79703489],
    4: [133.96612288, 3.85362098, 4.37284647],
    5: [217.97385828, 43.76697629, 229.43543734],
    6: [164.94309451, 125.22840326, 81.50394562],
    7: [157.10134385, 155.26893603, 193.22678809],
    8: [66.53429189, 32.62107138, 188.45387454],
    9: [157.58165138, 243.49941618, 159.97381151],
    10: [6.98127537, 5.22420501, 6.82420501],
    11: [48.88183862, 203.80514614, 203.66699975]
}


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=256, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)




    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):

        contracting_11_out = self.contracting_11(X)

        contracting_12_out = self.contracting_12(contracting_11_out)

        contracting_21_out = self.contracting_21(contracting_12_out)

        contracting_22_out = self.contracting_22(contracting_21_out)

        contracting_31_out = self.contracting_31(contracting_22_out)

        contracting_32_out = self.contracting_32(contracting_31_out)

        middle_out = self.middle(contracting_32_out)

        expansive_11_out = self.expansive_11(middle_out)

        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_31_out), dim=1))

        expansive_21_out = self.expansive_21(expansive_12_out)

        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_21_out), dim=1))

        expansive_31_out = self.expansive_31(expansive_22_out)

        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_11_out), dim=1))

        output_out = self.output(expansive_32_out)
        return output_out

model=Net(12)
model.load_state_dict(torch.load('./low_noise_model.pth', map_location = 'cpu'))

model_new = Net(12)
model_new.load_state_dict(torch.load('./high_noise_model.pth', map_location='cpu'))


h, w = 128, 128
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
])
n_colors = len(colors)
custom_cmap = LinearSegmentedColormap.from_list('custom', [colors[i] for i in range(n_colors)], N=n_colors)

def predict(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        model.eval()
        prediction = model(image).squeeze()
        prediction = torch.argmax(prediction, dim=0)
        prediction = (prediction-prediction.min()) / (prediction.max()-prediction.min())
        prediction = prediction.numpy()
    return prediction


st.title("Road Image Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, width=200, clamp=True)
    st.subheader("Prediction using Low noise model")
    if st.button("RUN", key=0):
        image = Image.open(uploaded_file)
        with st.spinner("Running..."):
            predictions = predict(image, model)
      
            ig= px.imshow(predictions)
            st.plotly_chart(ig)
            # st.image(ig, width=400)

    st.subheader("Prediction using High noise model")
    if st.button("RUN", key=1):
        image = Image.open(uploaded_file)
        with st.spinner("Running..."):
            predictions = predict(image, model_new)
      
            ig= px.imshow(predictions)
       
            st.plotly_chart(ig)