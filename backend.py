import ee

# Initialisation GEE avec votre ID de projet
try:
    # On force l'initialisation avec l'ID que vous avez trouvé
    ee.Initialize(project='non-commercial-471612')
except:
    ee.Authenticate()
    ee.Initialize(project='non-commercial-471612')

def run_analysis(roi_coords):
    # 1. Zone d'étude
    roi = ee.Geometry.Rectangle(roi_coords)
    
    # 2. Chargement Sentinel-2
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate('2023-01-01', '2023-12-31') \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    image = collection.median().clip(roi)
    
    # 3. Indices
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
    
    input_img = image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).addBands([ndvi, ndwi, ndbi])
    
    # 4. Points d'entraînement
    eau = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.98, 18.10]), {'class': 0})])
    vegetation = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.95, 18.08]), {'class': 1})])
    urbain = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-15.97, 18.09]), {'class': 2})])
    sable = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([-16.00, 18.12]), {'class': 3})])
    classes = eau.merge(vegetation).merge(urbain).merge(sable)
    
    # 5. Classification
    samples = input_img.sampleRegions(collection=classes, properties=['class'], scale=10)
    classifier = ee.Classifier.smileRandomForest(50).train(samples, 'class', input_img.bandNames())
    classified = input_img.classify(classifier)
    
    return classified