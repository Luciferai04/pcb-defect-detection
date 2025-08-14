Core API Reference
==================

.. automodule:: core.foundation_adapter
   :members:
   :undoc-members:
   :show-inheritance:

Foundation Adapter Classes
--------------------------

.. autoclass:: core.foundation_adapter.BaseFoundationAdapter
   :members:
   :special-members: __init__

.. autoclass:: core.foundation_adapter.CLIPAdapter
   :members:
   :special-members: __init__

LoRA Implementation
------------------

.. autoclass:: core.foundation_adapter.LoRALayer
   :members:
   :special-members: __init__

Domain Prompt Learning
---------------------

.. autoclass:: core.foundation_adapter.DomainPromptLearning
   :members:
   :special-members: __init__

Configuration
-------------

.. autoclass:: core.foundation_adapter.AdaptationConfig
   :members:

Training Framework
------------------

.. autoclass:: core.foundation_adapter.AdaptationTrainer
   :members:
   :special-members: __init__

Utility Functions
-----------------

.. autofunction:: core.foundation_adapter.create_foundation_adapter

.. autofunction:: core.foundation_adapter.set_reproducible_seed

.. autofunction:: core.foundation_adapter.mixed_precision_context
