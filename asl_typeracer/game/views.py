from django.shortcuts import render

def home(request):
    return render(request, "home.html")


def webcam_view(request):
    return render(request, "game/webcam.html")
