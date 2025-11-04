from django.db import models

# Create your models here.
class Prompt(models.Model):
    text = models.TextField()

class Score(models.Model):
    username = models.CharField(max_length = 50)
    prompt = models.ForeignKey(Prompt, on_delete=models.CASCADE)
    wpm = models.FloatField()
