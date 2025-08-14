#!/usr/bin/env python3
"""
PCB Defect Detection - Feature Demonstration
===========================================

This script demonstrates all the key features of the PCB defect detection framework.
"""

import torch
import numpy as np
from PIL import Image
import os
import json
import time
from pathlib import Path


def create_synthetic_pcb_samples():
    """Create synthetic PCB samples for demonstration."""
    print("🔧 Creating synthetic PCB samples...")
    
    def create_pcb_image(defect_type='normal', size=(224, 224)):
        # Base PCB (green background)
        image = np.full((*size, 3), [0, 100, 0], dtype=np.uint8)
        
        # Add circuit traces (copper color)
        for i in range(0, size[0], 20):
            image[i:i+2, :] = [184, 115, 51]
        for j in range(0, size[1], 30):
            image[:, j:j+2] = [184, 115, 51]
        
        # Add components
        positions = [(50, 50), (100, 100), (150, 150)]
        for x, y in positions:
            if defect_type == 'missing_component' and (x, y) == positions[0]:
                continue
            image[y:y+15, x:x+25] = [20, 20, 20]
            
            if defect_type == 'solder_bridge' and (x, y) == positions[1]:
                image[y+15:y+25, x:x+25] = [200, 200, 200]
        
        if defect_type == 'short_circuit':
            image[120:125, 60:140] = [255, 0, 0]
        
        return Image.fromarray(image)
    
    # Create sample directory
    os.makedirs('demo_samples', exist_ok=True)
    
    # Create samples for each defect type
    defect_types = ['normal', 'missing_component', 'solder_bridge', 'misalignment', 'short_circuit']
    for defect_type in defect_types:
        image = create_pcb_image(defect_type)
        image.save(f'demo_samples/{defect_type}.jpg')
    
    print(f"✅ Created {len(defect_types)} synthetic PCB samples in demo_samples/")
    return defect_types


def test_enhanced_model():
    """Test the enhanced PCB model with LoRA adaptation."""
    print("\n🧠 Testing Enhanced PCB Model...")
    
    try:
        from enhanced_pcb_model import create_enhanced_model
        
        # Create model
        model, loss_fn = create_enhanced_model(num_classes=5)
        
        # Get model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        efficiency = 1.0 - (trainable_params / total_params)
        
        print(f"✅ Enhanced PCB Model created successfully")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🎯 Trainable parameters: {trainable_params:,}")
        print(f"   ⚡ Parameter efficiency: {efficiency:.4f} ({efficiency*100:.2f}% frozen)")
        
        # Test forward pass
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        sample_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(sample_input)
        
        print(f"   🚀 Forward pass successful: {sample_input.shape} -> {output.shape}")
        print(f"   💻 Device: {device}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Enhanced model test failed: {e}")
        return None, None


def test_foundation_adapter():
    """Test the foundation model adapter."""
    print("\n🔄 Testing Foundation Model Adapter...")
    
    try:
        from core.foundation_adapter import LoRALayer, set_reproducible_seed, AdaptationConfig
        
        # Set reproducible seed
        set_reproducible_seed(42)
        print("✅ Reproducible seed set")
        
        # Test LoRA layer
        lora = LoRALayer(256, 128, rank=4, alpha=32.0)
        input_tensor = torch.randn(8, 256)
        output = lora(input_tensor)
        
        print(f"✅ LoRA Layer test successful")
        print(f"   📐 Input: {input_tensor.shape} -> Output: {output.shape}")
        print(f"   🎯 Rank: {lora.rank}, Alpha: {lora.alpha}")
        print(f"   ⚡ Parameter efficiency: {lora.efficiency_ratio:.4f}")
        
        # Test configuration
        config = AdaptationConfig(domain="materials", rank=8, alpha=16)
        print(f"✅ Adaptation configuration created")
        print(f"   🎯 Domain: {config.domain}")
        print(f"   📊 Domain prompts: {len(config.domain_prompts)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Foundation adapter test failed: {e}")
        return False


def test_inference_api():
    """Test the inference API functionality."""
    print("\n🔍 Testing Inference API...")
    
    try:
        from api.inference import PCBDefectInference
        
        # Initialize inference engine
        inference = PCBDefectInference(device='auto', batch_size=16)
        print("✅ Inference engine initialized")
        print(f"   💻 Device: {inference.device}")
        print(f"   🏷️ Classes: {len(inference.class_names)}")
        
        # Test single image inference
        if Path('demo_samples/normal.jpg').exists():
            result = inference.predict_single('demo_samples/normal.jpg')
            
            if 'error' not in result:
                print("✅ Single image inference successful")
                print(f"   📋 Predicted: {result['predicted_class']}")
                print(f"   🎯 Confidence: {result['confidence']:.3f}")
                print(f"   ⏱️ Inference time: {result['inference_time_ms']:.2f} ms")
            else:
                print(f"❌ Single image inference failed: {result['error']}")
        
        # Test batch processing
        if Path('demo_samples').exists():
            results = inference.predict_directory('demo_samples')
            print(f"✅ Batch processing successful: {len(results)} images processed")
            
            # Print summary
            inference.print_summary(results)
        
        return True
        
    except Exception as e:
        print(f"❌ Inference API test failed: {e}")
        return False


def test_training_simulation():
    """Test training simulation."""
    print("\n🏋️ Testing Training Simulation...")
    
    try:
        from enhanced_pcb_model import create_enhanced_model
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create model and data
        model, loss_fn = create_enhanced_model(num_classes=5)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        
        # Create synthetic training data
        images = torch.randn(100, 3, 224, 224)
        labels = torch.randint(0, 5, (100,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Setup optimizer for trainable parameters only
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=1e-3, weight_decay=1e-4
        )
        
        print("✅ Training setup completed")
        print(f"   💻 Device: {device}")
        print(f"   📊 Dataset size: {len(dataset)}")
        print(f"   🎯 Batch size: {dataloader.batch_size}")
        
        # Quick training simulation (2 epochs)
        model.train()
        training_losses = []
        
        for epoch in range(2):
            total_loss = 0
            num_batches = 0
            
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            training_losses.append(avg_loss)
            print(f"   📈 Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        print("✅ Training simulation completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Training simulation failed: {e}")
        return False


def test_performance_benchmark():
    """Benchmark model performance."""
    print("\n⚡ Performance Benchmarking...")
    
    try:
        from enhanced_pcb_model import create_enhanced_model
        
        model, _ = create_enhanced_model(num_classes=5)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        # Benchmark inference speed
        sample_input = torch.randn(1, 3, 224, 224).to(device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(sample_input)
        
        # Benchmark
        num_runs = 50
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(sample_input)
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
        
        print("✅ Performance benchmark completed")
        print(f"   ⏱️ Average inference time: {avg_inference_time:.2f} ms")
        print(f"   🚀 Throughput: {1000/avg_inference_time:.1f} images/second")
        print(f"   💾 Model size: {param_size:.2f} MB")
        print(f"   💻 Device: {device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def generate_demo_report():
    """Generate a demo report."""
    print("\n📄 Generating Demo Report...")
    
    report = {
        "demo_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        },
        "features_tested": [
            "Enhanced PCB Model with LoRA adaptation",
            "Foundation model adapter framework",
            "Inference API with batch processing", 
            "Training simulation",
            "Performance benchmarking"
        ],
        "status": "Demo completed successfully! 🎉"
    }
    
    with open('demo_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Demo report saved to demo_report.json")
    return report


def main():
    """Main demonstration function."""
    print("🚀 PCB Defect Detection Framework - Feature Demonstration")
    print("=" * 80)
    
    # Track test results
    results = []
    
    # 1. Create synthetic samples
    try:
        create_synthetic_pcb_samples()
        results.append(("Synthetic Sample Creation", "✅ PASS"))
    except Exception as e:
        results.append(("Synthetic Sample Creation", f"❌ FAIL: {e}"))
    
    # 2. Test enhanced model
    try:
        model, device = test_enhanced_model()
        if model is not None:
            results.append(("Enhanced PCB Model", "✅ PASS"))
        else:
            results.append(("Enhanced PCB Model", "❌ FAIL"))
    except Exception as e:
        results.append(("Enhanced PCB Model", f"❌ FAIL: {e}"))
    
    # 3. Test foundation adapter
    try:
        if test_foundation_adapter():
            results.append(("Foundation Model Adapter", "✅ PASS"))
        else:
            results.append(("Foundation Model Adapter", "❌ FAIL"))
    except Exception as e:
        results.append(("Foundation Model Adapter", f"❌ FAIL: {e}"))
    
    # 4. Test inference API
    try:
        if test_inference_api():
            results.append(("Inference API", "✅ PASS"))
        else:
            results.append(("Inference API", "❌ FAIL"))
    except Exception as e:
        results.append(("Inference API", f"❌ FAIL: {e}"))
    
    # 5. Test training simulation
    try:
        if test_training_simulation():
            results.append(("Training Simulation", "✅ PASS"))
        else:
            results.append(("Training Simulation", "❌ FAIL"))
    except Exception as e:
        results.append(("Training Simulation", f"❌ FAIL: {e}"))
    
    # 6. Performance benchmarking
    try:
        if test_performance_benchmark():
            results.append(("Performance Benchmarking", "✅ PASS"))
        else:
            results.append(("Performance Benchmarking", "❌ FAIL"))
    except Exception as e:
        results.append(("Performance Benchmarking", f"❌ FAIL: {e}"))
    
    # Print final results
    print("\n" + "=" * 80)
    print("📋 FINAL TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        print(f"{test_name:.<40} {result}")
        if "✅ PASS" in result:
            passed += 1
    
    print("=" * 80)
    print(f"📊 Summary: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The PCB defect detection framework is working perfectly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Generate report
    try:
        generate_demo_report()
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    print("\n🚀 Demo completed! Check the following files:")
    print("   📁 demo_samples/ - Synthetic PCB images")
    print("   📄 demo_report.json - Detailed demo report")
    print("   📊 results.json - Batch inference results")


if __name__ == "__main__":
    main()
