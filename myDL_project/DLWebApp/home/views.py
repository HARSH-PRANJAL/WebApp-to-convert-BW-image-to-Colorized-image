from django.shortcuts import render, redirect, HttpResponse
from .models import GrayscaleImage, ColorizedImage, ResizedImage
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.conf import settings
from django.views import View
from PIL import Image
import numpy as np
import uuid
import io
import cv2
import os

NETWORK = os.path.join(settings.BASE_DIR, r"DL/colorization_deploy_v2.prototxt")
VALS = os.path.join(settings.BASE_DIR, r"DL/pts_in_hull.npy")
MODEL = os.path.join(settings.BASE_DIR, r"DL/colorization_release_v2.caffemodel")

# Load the model for colorizing
print("Loading model")
net = cv2.dnn.readNetFromCaffe(NETWORK, MODEL)
pts = np.load(VALS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]


def colorize_image(image_path):
    image = cv2.imread(image_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized


def resize_image(image, size=(224, 224)):
    try:
        resized_image = image.resize(size)
        return resized_image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class Index(View):

    def get(self, request):
        return render(request, "upload_form.html")

    def post(self, request):
        image = request.FILES["greyImage"]
        if not image:
            return HttpResponse("no image")

        # Get the uploaded file
        image_file = request.FILES["greyImage"]

        # Save the grayscale image
        grayscale_image = GrayscaleImage(image=image_file)
        grayscale_image.save()

        # Open the uploaded image with Pillow
        pil_image = Image.open(image_file)

        # Resize the image
        resized_image = resize_image(pil_image)
        if resized_image is None:
            return HttpResponse("Error resizing image", status=500)

        # Save the resized image to a BytesIO object
        resized_image_io = io.BytesIO()
        resized_image.save(resized_image_io, format="JPEG")
        resized_image_file = ContentFile(
            resized_image_io.getvalue(), "resized_image.jpg"
        )

        # Save the resized image to the database
        resized_image_entry = ResizedImage(
            grayscale_image=grayscale_image, image=resized_image_file
        )
        resized_image_entry.save()

        # temperary path
        temp_path = default_storage.save(
            "temp/" + str(uuid.uuid4()) + ".jpg",
            ContentFile(resized_image_io.getvalue()),
        )

        # call the funtion to convert grey scale image to coloured image
        colorized_image_np = colorize_image(default_storage.path(temp_path))
        colorized_pil_image = Image.fromarray(colorized_image_np)
        colorized_image_io = io.BytesIO()
        colorized_pil_image.save(colorized_image_io, format="JPEG")
        colorized_image_file = ContentFile(
            colorized_image_io.getvalue(), "colorized_image.jpg"
        )

        # Save the colorized image to the database
        colorized_image_entry = ColorizedImage(
            grayscale_image=grayscale_image, image=colorized_image_file
        )
        colorized_image_entry.save()
        response_data = {
            "signature_token": grayscale_image.signature,
            "download_token": colorized_image_entry.download_token,
        }

        return JsonResponse(response_data, status=201)


class Download(View):

    def get(self,request):
        return render(request,"download_form.html")
    
    def post(self,request):
        token = request.POST.get("token")
        if not token:
            return HttpResponse("No token provided")

        # Fetch the colorized image using the token
        colorized_image = ColorizedImage.objects.filter(download_token=token).first()
        if not colorized_image:
            return HttpResponse("No image found for this token")

        try:
            # Open and return the colorized image
            with default_storage.open(colorized_image.image.name, "rb") as f:
                response = HttpResponse(f.read(), content_type="image/jpeg")
                response["Content-Disposition"] = (
                    f'attachment; filename="colorized_image.jpg"'
                )
                return response
        except Exception as e:
            return HttpResponse(f"Error during download: {str(e)}")