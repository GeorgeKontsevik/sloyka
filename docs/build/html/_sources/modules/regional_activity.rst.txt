.. _regional_activity:

Regional activity
==================
The regional_activity module is designed to aggregate data by region and provide information about user activity.
The RegionalActivity class creates a GeoDataFrame with basic information about user activity, using other modules such as geocoder,
text classifier, city_services_extract and emotion_classifier to process the data. The processed data is stored in the class attribute
processed_geodata and can be called after the class is initialized with RegionalActivity.processed_geodata. The class includes the get_risks()
function, which returns a DataFrame with social risk information based on the provided texts.