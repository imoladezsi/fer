import os

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render

from emotion_recognition.Backend import Backend
from .const import BASE_PATH

from .Utils import Utils
from .forms import PredictionForm


def index(request):
    form = PredictionForm()
    if request.method == 'POST' and len(request.FILES) > 0:
        # get all the information from the UI
        file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(file.name, file)
        uploaded_file_url = fs.url(filename)
        estimator_path = request.POST['predictor']
        image_path = os.path.join(BASE_PATH,uploaded_file_url[1:])
        print(image_path)
        backend = Backend(estimator_path)

        prediction_result = backend.predict(image_path)

        return render(request, "index.html", {'image': image_path,'form': form, 'max': Utils.get_max(prediction_result) , 'prediction':
                                                            zip(prediction_result['emotions'],
                                                                [format(e * 100, '2.2f')
                                                                for e in prediction_result['result']])})
    return render(request, "index.html", {'form': form})
