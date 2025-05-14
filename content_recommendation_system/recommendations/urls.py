from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('matches/', views.matches, name='matches'),
    path('recommend/', views.recommend, name='recommend'),
    path('top/', views.top_movies, name='top_movies'),

]