from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import pickle
import numpy as np 
import pandas as pd
from django.templatetags.static import static

# Create your views here.
from .forms import RegisterForm

def index(request):
    return render(request, 'accounts/index.html', {})

def registerUser(request):
    form = RegisterForm()
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')

    else:
        form = RegisterForm()
    return render(request, 'accounts/register.html', {'form':form})

def loginUser(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username and password:

            user = authenticate(username=username, password=password)

            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.error(request, 'Username or Password is Incorrect')
        else:
            messages.error(request, 'Fill out all the fields')

    return render(request, 'accounts/login.html', {})

def home(request):
    context = {}
    if request.POST:
        Pregencies = float(request.POST.get('Pregencies'))
        Glucose = float(request.POST.get('Glucose'))
        BloodPressure = float(request.POST.get('BloodPressure'))
        SkinThickness = float(request.POST.get('SkinThickness'))
        Insulin = float(request.POST.get('Insulin'))
        BMI = float(request.POST.get('BMI'))
        DiabetesPedigreeFunction = float(request.POST.get('DiabetesPedigreeFunction'))
        Age = float(request.POST.get('Age'))
        context = {'Pregencies':[Pregencies],
        'Glucose':[Glucose],
        'BloodPressure':[BloodPressure],
        'SkinThickness':[SkinThickness],
        'Insulin':[Insulin],
        'BMI':[BMI],
        'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
        'Age':[Age]
        }

        data = pd.DataFrame.from_dict(context)
        res = ''
        with open('GNB.pkl', 'rb') as f:
            GNB = pickle.load(f)
            res = GNB.predict(data)
            res_prob = GNB.predict_log_proba(data)
        predict = ''
        if res ==1:
            predict = '+ve'
        else:
            predict= '-ve'
        context.update({'res':predict, 'res_prob':res_prob})

    return render(request, 'accounts/home.html', context)


def logoutUser(request):
    logout(request)
    return redirect('index')