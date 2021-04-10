# Generated by Django 3.1.6 on 2021-02-13 03:58

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('accounts', '0003_delete_patientinfo'),
    ]

    operations = [
        migrations.CreateModel(
            name='patientInfo',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Pregencies', models.IntegerField()),
                ('Glucose', models.IntegerField()),
                ('BloodPressure', models.IntegerField()),
                ('SkinThickness', models.IntegerField()),
                ('Insulin', models.IntegerField()),
                ('BMI', models.IntegerField()),
                ('Age', models.IntegerField()),
                ('DiabetesPedigreeFunction', models.IntegerField()),
                ('result', models.CharField(max_length=100)),
                ('username', models.CharField(max_length=100)),
            ],
        ),
    ]