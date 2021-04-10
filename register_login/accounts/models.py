from django.db import models


# class PatientInfo(models.Model):
#     Pregencies =models.DecimalField(..., max_digits=5, decimal_places=2)
#     Glucose = models.DecimalField(..., max_digits=5, decimal_places=2)
#     BloodPressure = models.DecimalField(..., max_digits=5, decimal_places=2)
#     SkinThickness = models.DecimalField(..., max_digits=5, decimal_places=2)
#     Insulin = models.DecimalField(..., max_digits=5, decimal_places=2)
#     BMI = models.DecimalField(..., max_digits=5, decimal_places=2)
#     Age = models.DecimalField(..., max_digits=5, decimal_places=2)
#     DiabetesPedigreeFunction = models.DecimalField(..., max_digits=5, decimal_places=2)
#     result = models.CharField(max_length=100)
#     username = models.CharField(max_length=100)

    
class patientInfo(models.Model):
    Pregencies = models.IntegerField()
    Glucose = models.IntegerField()
    BloodPressure = models.IntegerField()
    SkinThickness = models.IntegerField()
    Insulin = models.IntegerField()
    BMI = models.IntegerField()
    Age = models.IntegerField()
    DiabetesPedigreeFunction = models.IntegerField()
    result = models.CharField(max_length=100)
    username = models.CharField(max_length=100) 
