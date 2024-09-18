from django.urls import path
from .views import Index,Download

urlpatterns = [
    path("", Index.as_view(), name="index"),
    path("upload_grey_image", Index.as_view(), name="upload_grey_image"),
    path("download_coloured_image", Download.as_view(), name="download_coloured_image"),
]
