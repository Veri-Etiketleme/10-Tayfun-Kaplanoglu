# Generated by Django 2.1.4 on 2019-02-04 13:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_auto_20190128_1428'),
    ]

    operations = [
        migrations.AlterField(
            model_name='documentprocesssettings',
            name='text_direction',
            field=models.CharField(choices=[('horizontal-lr', 'Horizontal l2r'), ('horizontal-rl', 'Horizontal r2l'), ('vertical-lr', 'Vertical l2r'), ('vertical-rl', 'Vertical r2l')], default='vertical-lr', max_length=64),
        ),
    ]
