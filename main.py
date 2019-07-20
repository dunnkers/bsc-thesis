import fire
from src.cache import cache_all
from src.transform import transform_all
from src.train import train_all
from src.test import test_all

class RoadMarkerClassification(object):
  """ Road Marker Classification."""

  def cache(self):
    cache_all()

  def transform(self):
    transform_all()

  def train(self):
    train_all()

  def test(self):
    test_all()

if __name__ == '__main__':
  fire.Fire(RoadMarkerClassification)

