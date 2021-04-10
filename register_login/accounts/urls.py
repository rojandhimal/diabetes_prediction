from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('register/', views.registerUser, name='register'),
    path('login/', views.loginUser, name='login'),
    path('logout/', views.logoutUser, name='logout'),
    path("viewreport/", views.ViewPDF.as_view(), name='viewreport'),
    path('downloadreport/', views.DownloadPDF.as_view(), name='downloadreport'),
    path('pdf_view/', views.ViewPDF.as_view(), name="pdf_view"),
    path('pdf_download/', views.DownloadPDF.as_view(), name="pdf_download"),
    path('traindata/', views.trainData, name="traindata"),

]
