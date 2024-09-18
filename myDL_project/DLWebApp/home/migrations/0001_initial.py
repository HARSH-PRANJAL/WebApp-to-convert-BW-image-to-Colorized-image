# Generated by Django 5.1.1 on 2024-09-18 01:03

import django.db.models.deletion
import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GrayscaleImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='grayscale_images/')),
                ('signature', models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='ColorizedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='colorized_images/')),
                ('download_token', models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ('grayscale_image', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='home.grayscaleimage')),
            ],
        ),
        migrations.CreateModel(
            name='ResizedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='resized_images/')),
                ('grayscale_image', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='home.grayscaleimage')),
            ],
        ),
    ]
