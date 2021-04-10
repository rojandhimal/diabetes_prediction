from django.contrib import admin

from .models import patientInfo

# Register your models here.


class PatientAdmin(admin.ModelAdmin):
    search_fields = ('username',)


admin.site.register(patientInfo, PatientAdmin)
