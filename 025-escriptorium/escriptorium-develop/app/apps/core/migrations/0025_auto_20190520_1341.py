# Generated by Django 2.1.4 on 2019-05-20 13:41

import core.models
import django.contrib.postgres.fields.jsonb
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0024_auto_20190520_1138'),
    ]

    operations = [
        migrations.AddField(
            model_name='ocrmodel',
            name='revision',
            field=models.UUIDField(default=uuid.uuid4, editable=False),
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='training_accuracy',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='training_errors',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='training_total',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='version_author',
            field=models.CharField(default='unknown', editable=False, max_length=128),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='version_created_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='version_source',
            field=models.CharField(default='escriptorium', editable=False, max_length=128),
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='version_updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
        migrations.AddField(
            model_name='ocrmodel',
            name='versions',
            field=django.contrib.postgres.fields.jsonb.JSONField(default=list, editable=False),
        ),
        migrations.AlterField(
            model_name='ocrmodel',
            name='document',
            field=models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='ocr_models', to='core.Document'),
        ),
        migrations.AlterField(
            model_name='ocrmodel',
            name='file',
            field=models.FileField(upload_to=core.models.models_path, validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['mlmodel'])]),
        ),
    ]
