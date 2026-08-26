Reference
=========

Model identity
--------------

A SynDE energy prediction is a linear sum of an intercept, elemental
composition terms, and named connectivity terms:

.. math::

   \hat E(G) = b + \sum_e n_e(G) w_e
   + \sum_j x_j(G) \beta_j.

For constitutional isomers, the elemental counts :math:`n_e` are identical.
Their ordering therefore depends only on the connectivity block. A
connectivity contribution is an attributable model term, not an
electronic-structure energy decomposition.

Artifact and validation terms
-----------------------------

``model card``
   Declares the target protocol, units, training source, chemical domain, and
   evaluation status of an artifact.

``model_sha256``
   Digest of the exact JSON artifact loaded for inference.

``feature distance``
   Descriptive standardized distance from training feature statistics. It is
   a warning boundary, not a probability or uncertainty interval.

``external refinement``
   A ridge-anchored update fitted to user-supplied labels. The resulting model
   requires a new independent validation and does not inherit the packaged
   artifact's evaluation claim.

Reproducibility checks
----------------------

Run the same local gates used for development:

.. code-block:: console

   conda activate synde
   bash scripts/lint.sh
   bash scripts/pytest.sh -q
   bash scripts/build_doc.sh

Package artifacts can be checked after building:

.. code-block:: console

   python -m build
   python scripts/check_package_artifacts.py dist

.. bibliography:: refs.bib
   :style: unsrt
   :cited:

License
-------

SynDE is distributed under the MIT License. See the `LICENSE file
<https://github.com/TieuLongPhan/SynDE/blob/main/LICENSE>`_.

Acknowledgments
---------------

This project received funding from the European Union's Horizon Europe
Doctoral Network programme under Marie Skłodowska-Curie grant agreement No.
101072930 (`TACsy <https://tacsy.eu/>`_).
