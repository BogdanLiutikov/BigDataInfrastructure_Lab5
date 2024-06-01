from kmean import KMean
from logger import Logger

logger = Logger(True).get_logger(__name__)

kmean = KMean()

data = kmean.load_data('data/en.openfoodfacts.org.products.csv', 1_000)

feature_data = kmean.make_feature_column(data)
scaled_data = kmean.standard_scale(feature_data)
kmean.train(scaled_data)

pred = kmean.predict(scaled_data)
pred.select('features', 'prediction').sample(0.01).show()

kmean.plot(pred)

eval = kmean.eval(pred)
logger.info(f'Eval\n {eval}')
