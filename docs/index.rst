PCB Defect Detection Research Documentation
===========================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/pytorch-2.0%2B-orange
   :target: https://pytorch.org/
   :alt: PyTorch Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://opensource.org/licenses/MIT
   :alt: License

Welcome to the comprehensive documentation for the **PCB Defect Detection Research** project, a state-of-the-art foundation model adaptation framework for industrial quality control.

🎯 **Research Achievements**

- **90.5% accuracy** on PCB defect classification
- **98.8% parameter efficiency** (only 1.76% trainable parameters)
- **Real-time inference** capability for industrial deployment
- **Cross-domain validation** across 4+ specialized domains

🔬 **Key Innovations**

- **LoRA Adaptation**: Parameter-efficient fine-tuning with <2% trainable parameters
- **Multi-Scale Pyramid Attention**: Novel attention mechanism for fine-grained defect detection
- **Progressive Domain Adaptation**: 4-stage curriculum learning approach
- **Enhanced Active Learning**: Intelligent sample selection for data-scarce environments

Quick Start
-----------

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt
   
   # Run inference
   python -m api.inference --model enhanced_pcb_model.pth --image sample.jpg
   
   # Start training
   python train_active_learning.py --config config.py

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/foundation_models
   user_guide/active_learning
   user_guide/model_training
   user_guide/inference
   user_guide/evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/models
   api/training
   api/evaluation
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Research

   research/methodology
   research/experiments
   research/results
   research/publications

.. toctree::
   :maxdepth: 2
   :caption: Deployment

   deployment/docker
   deployment/production
   deployment/monitoring
   deployment/troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
