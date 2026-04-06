Data Sources
============

DEMOS reads all input data from flat files declared in the ``[[tables]]`` section of your configuration
file. Every entry in ``[[tables]]`` describes one table that will be loaded into memory and made
available to the simulation by name. DEMOS supports two file formats: **CSV** and **HDF5 (H5)**.

This page explains when to use each format, what each parameter controls, and shows complete
configuration examples for both.

----

Choosing Between CSV and HDF5
-----------------------------

The two formats serve different roles and you will often use both in the same project:

* **CSV files** are plain-text tables, one row per line, columns separated by commas (or another
  delimiter you specify). They are easy to open in a spreadsheet application, easy to edit, and
  easy to produce from any tool. Use CSV for smaller or auxiliary tables such as calibration
  reference data, mapping tables, and control totals.

* **HDF5 files** (``.h5``) are a binary format designed to store large numerical datasets
  efficiently. They can contain multiple tables (called *keys*) in a single file. Use H5 for
  the main synthetic population tables (``persons`` and ``households``) because they are
  typically large and the binary format reads much faster than CSV.

----

CSV Tables
----------

A CSV entry in the ``[[tables]]`` section looks like this:

.. code-block:: toml

   [[tables]]
   file_type = "csv"
   table_name = "income_rates"
   filepath = "../data/my_region/income_rates.csv"
   index_col = "year"

The ``file_type = "csv"`` value identifies this entry as a CSV source.

Required parameters
~~~~~~~~~~~~~~~~~~~

* ``file_type`` — Must be ``"csv"``.
* ``table_name`` — The name by which this table will be referred to everywhere in DEMOS
  (for example, in module configuration entries like ``observed_values_table``).
  Choose a short, descriptive name with no spaces, using underscores: e.g.
  ``"observed_births_data"`` or ``"income_rates"``.
* ``filepath`` — Path to the CSV file. This can be a relative path (evaluated relative to the
  configuration file location) or an absolute path.
* ``index_col`` — The column in the CSV file to use as the row identifier (the *index*).
  This column will not appear as a regular data column; it becomes the label for each row.
  For calibration tables the index must be ``"year"``.

Optional parameters
~~~~~~~~~~~~~~~~~~~

* ``delimiter`` — The character used to separate columns. Defaults to ``","`` for standard CSV.
  Set to ``"\t"`` for tab-separated files or ``";"`` for semicolon-separated files.
* ``custom_dtype_casting`` — An inline mapping from column names to data types. This is useful
  when pandas would otherwise infer the wrong type, such as reading a GEOID code as an integer
  when it should be kept as a string. Example:

  .. code-block:: toml

     custom_dtype_casting = {"lcm_county_id" = "object", "year" = "int", "rate" = "float"}

  Supported type strings are any value accepted by ``pandas.read_csv``'s ``dtype`` argument:
  ``"object"`` (string), ``"int"``, ``"float"``, ``"bool"``.

Full example with all parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [[tables]]
   file_type = "csv"
   table_name = "income_rates"
   filepath = "../data/my_region/observed_calibration_values/income_rates.csv"
   index_col = "year"
   custom_dtype_casting = {"lcm_county_id" = "object", "year" = "int", "rate" = "float"}

----

HDF5 Tables
-----------

An HDF5 entry in the ``[[tables]]`` section looks like this:

.. code-block:: toml

   [[tables]]
   file_type = "h5"
   table_name = "persons"
   filepath = "../data/my_region/population.h5"
   h5_key = "persons"

The ``file_type = "h5"`` value identifies this entry as an HDF5 source.

Required parameters
~~~~~~~~~~~~~~~~~~~

* ``file_type`` — Must be ``"h5"``.
* ``table_name`` — The name by which this table will be referred to everywhere in DEMOS.
  For the main synthetic population this is almost always ``"persons"`` and ``"households"``.
* ``filepath`` — Path to the HDF5 file.
* ``h5_key`` — The key inside the HDF5 file that identifies the table to load. An HDF5 file
  can contain many tables under different keys, similar to sheets in a spreadsheet workbook.
  To inspect the keys in an HDF5 file you can open it in Python:

  .. code-block:: python

     import pandas as pd
     store = pd.HDFStore("../data/my_region/population.h5")
     print(store.keys())   # e.g. ['/persons', '/households']
     store.close()

Full example with all parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   [[tables]]
   file_type = "h5"
   table_name = "persons"
   filepath = "../data/my_region/population.h5"
   h5_key = "persons"

   [[tables]]
   file_type = "h5"
   table_name = "households"
   filepath = "../data/my_region/population.h5"
   h5_key = "households"

Note that two tables can be loaded from the same HDF5 file by using two separate
``[[tables]]`` entries that share the same ``filepath`` but point to different ``h5_key`` values.

----

API Reference
-------------

.. autoclass:: demos.datasources.CSVTableSource
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: load_into_orca

.. autoclass:: demos.datasources.H5TableSource
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: load_into_orca