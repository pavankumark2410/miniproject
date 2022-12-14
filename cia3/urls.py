from .views import algorithm
from django.urls import re_path as url
from django.urls import path
from cia3 import views
urlpatterns = [
path('',views.algorithm,name='prediction'),
]