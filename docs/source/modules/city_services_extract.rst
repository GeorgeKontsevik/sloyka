.. _city_services:

Services extraction
==========================
The City_services class is designed to extract city service names from text using a string comparison algorithm, taking into account
 the changing service endings in the text. Using the flair library, the City_services.run() method in messages extracts named entities
  from the Sentence object as a list, as well as the most probable service type, and stores them in new columns of the original DataFrame().
.. automodule:: sloyka.src.utils.data_processing.city_services_extract
   :members:
   :undoc-members:
   