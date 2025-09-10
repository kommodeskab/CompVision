# Setup
Download data mappen 'hotdog_nothotdog' og hiv ind i git repository. 
Installér de nødvendige pakker:
`pip install pytorch-lightning` og `pip install torchvision`.

Lav en ny branch med dit navn. Så kan man skubbe så mange mærkelige ændringer, man vil. Vi merger branches til sidst.

Kør `main.py` for at lave et run. Tjek resultater af et run ved at taste: `tensorboard --logdir=lightning_logs/`.

# Arbejdsfordeling
* Rasmus: network architecture. Se fx på Resnet. Under transfer-learning, i.e. pretrained modeller.
* Felix: data (transforms) + visualization.py (visualisering af billeder)
* Nikolaj: "Compute the saliency map"  og "smoothgrad saliency map", prøv at plotte det, etc. 
* Gustav: Kig på metrics. fx lav et Callback (pl.Callback) som beregner accuracy. Undersøg hvilke billeder bliver misclassified.
* Andreas: tech support