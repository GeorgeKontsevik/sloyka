Main pipeline
============

By selecting a limited urbanized area and a list of online communities in a social network,
it is possible to run this dataset across all major library functions. However, in some cases,
the order in which they are run is important.

.. figure:: /image/etap.png
   :align: center
   :alt: photo

   SLOYKA's sections

The main sections were divided into:
* Data receiving (a step possible to skip only if there is already geolocated
text data mentioning urban sites, otherwise the steps are very important - :ref:`data_getter` and :ref:`geocoder` )

* Data tagging: Characterization of messages and urban objects, which can be carried out in any order: :ref:`emotion_classifier` :ref:`text_classifier` :ref:`city_services` :ref:`topic_modeler`

* Data modelling: Section consists of further synthesis of the obtained data, risk assessment and forecasting.
Each of the methods in this group requires certain labeling columns: :ref:`sem_graph` :ref:`regional_activity`

* Data visualization: The last step is applied to the already generated semantic graph - :ref:`graph_visualization`

You can get more info about each step in!
