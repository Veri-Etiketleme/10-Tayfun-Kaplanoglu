# Generated by Django 2.2.20 on 2021-05-20 13:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0043_auto_20210324_1016'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='read_direction',
            field=models.CharField(choices=[('ltr', 'Left to right'), ('rtl', 'Right to left')], default='ltr', help_text='The read direction describes the order of the elements in the document, in opposition with the text direction which describes the order of the words in a line and is set by the script.', max_length=3),
        ),
    ]
