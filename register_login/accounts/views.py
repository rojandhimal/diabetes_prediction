from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import pickle
import numpy as np
import pandas as pd
from django.templatetags.static import static
from django.views import View
from io import BytesIO
from django.http import HttpResponse
from django.template.loader import get_template
from xhtml2pdf import pisa

from .models import patientInfo

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


# Create your views here.
from .forms import RegisterForm


def trainModel(data):
    test = data.iloc[700:]
    test_label = test["Label"]
    test.drop(labels=["Label"], axis=1, inplace=True)

    train_df = data.iloc[:700]
    X_train = train_df.drop(labels=["Label"], axis=1)
    Y_train = train_df["Label"]

    x_train, x_test, y_train, y_test = train_test_split(
        X_train, Y_train, test_size=0.2, random_state=42)

    clf = GaussianNB()
    clf.fit(x_train, y_train)
    acc_clf_train = round(clf.score(x_train, y_train)*100, 2)
    acc_clf_test = round(clf.score(x_test, y_test)*100, 2)
    print("GaussianNB Train Accuracy: %", acc_clf_train)
    print("GaussianNB Test Accuracy: %", acc_clf_test)
    return clf, test, test_label


def testModel(clf, testdata, test_label):
    prediction = clf.predict(testdata)
    accuracy = accuracy_score(prediction, test_label)
    return accuracy


def saveModel(cfl):
    with open('my_dumped_classifier.pkl', 'wb') as fid:
        pickle.dump(gnb, fid)


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
    return render(request, 'accounts/register.html', {'form': form})


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


@login_required(login_url='login')  # redirect when user is not logged in
def home(request):
    context = {}
    if request.POST:
        Pregencies = float(request.POST.get('Pregencies'))
        Glucose = float(request.POST.get('Glucose'))
        BloodPressure = float(request.POST.get('BloodPressure'))
        SkinThickness = float(request.POST.get('SkinThickness'))
        Insulin = float(request.POST.get('Insulin'))
        BMI = float(request.POST.get('BMI'))
        DiabetesPedigreeFunction = float(
            request.POST.get('DiabetesPedigreeFunction'))
        Age = float(request.POST.get('Age'))
        context = {'Pregencies': [Pregencies],
                   'Glucose': [Glucose],
                   'BloodPressure': [BloodPressure],
                   'SkinThickness': [SkinThickness],
                   'Insulin': [Insulin],
                   'BMI': [BMI],
                   'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
                   'Age': [Age]
                   }

        data = pd.DataFrame.from_dict(context)
        res = ''
        with open('GNB.pkl', 'rb') as f:
            GNB = pickle.load(f)
            res = GNB.predict(data)
            res_prob = GNB.predict_log_proba(data)
        predict = ''
        if res == 1:
            predict = '+ve'
        else:
            predict = '-ve'
        context.update({'res': predict, 'res_prob': res_prob})
        p = patientInfo(Pregencies=Pregencies, Glucose=Glucose, BloodPressure=BloodPressure, SkinThickness=SkinThickness,
                        Insulin=Insulin, BMI=BMI, DiabetesPedigreeFunction=DiabetesPedigreeFunction, Age=Age, username=request.user, result=predict)
        p.save()

    return render(request, 'accounts/home.html', context)


def logoutUser(request):
    logout(request)
    return redirect('index')


# this section is for pdf
def render_to_pdf(template_src, context_dict={}):
	template = get_template(template_src)
	html = template.render(context_dict)
	result = BytesIO()
	pdf = pisa.pisaDocument(BytesIO(html.encode("ISO-8859-1")), result)
	if not pdf.err:
		return HttpResponse(result.getvalue(), content_type='application/pdf')
	return None


data = {
	"company": "Diabetis Prediction System",
	"address": "Kathmandu",
	"city": "Kathmandu",
	"state": "03",
	"zipcode": "44600",
	"phone": "+977-9823043456",
	"email": "milandhimal13@gmail.com.com",
	"website": "Diabetis Prediction System",

}

#Opens up page as PDF


class ViewPDF(View):
    def get(self, request, *args, **kwargs):
        pdf = render_to_pdf('accounts/pdf_template.html', data)
        return HttpResponse(pdf, content_type='application/pdf')


#Automaticly downloads to PDF file
class DownloadPDF(View):
    def getdbObj(self):
        patinfo = patientInfo.objects.last()
        return patinfo

    def get(self, request, *args, **kwargs):
        pinfo = self.getdbObj()
        data.update({'Pregencies': pinfo.Pregencies, 'Glucose': pinfo.Glucose, 'BloodPressure': pinfo.BloodPressure, 'SkinThickness': pinfo.SkinThickness, 'Insulin': pinfo.Insulin,
                    'BMI': pinfo.BMI, 'DiabetesPedigreeFunction': pinfo.DiabetesPedigreeFunction, 'Age': pinfo.Age, 'username': request.user, 'result': pinfo.result})
        pdf = render_to_pdf('accounts/pdf_template.html', data)
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = "Invoice_%s.pdf" % ("12341231")
        content = "attachment; filename=%s" % (filename)
        response['Content-Disposition'] = content
        return response


def trainData(request):
    print("this is train page")
    if request.POST:
        print("Train page btn clicked")
        f1 = request.POST['dataset1']
        f2 = request.POST['dataset2']
        f3 = request.POST['dataset3']
        m1, t1, tl1 = trainModel(f1)
        m2, t2, tl2 = trainModel(f2)
        m3, t3, tl3 = trainModel(f3)
        ac1 = testModel(m1, t1, tl1)
        ac2 = testModel(m2, t2, tl2)
        ac3 = testModel(m3, t3, tl3)
        if ac1 > ac2 and ac1 > ac3:
            saveModel(m1)
        elif ac2 > ac1 and ac2 > ac3:
            saveModel(m2)
        else:
            saveModel(m3)

    if request.GET:
        print("back to admin")
    return render(request, 'accounts/train.html', {})
