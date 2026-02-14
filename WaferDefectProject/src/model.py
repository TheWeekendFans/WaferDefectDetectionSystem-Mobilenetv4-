import torch
import torch.nn as nn
import timm

def get_mobilenetv4(num_classes=38, pretrained=False):
    """
    Creates MobileNetV4 model.
    Using 'mobilenetv4_conv_small' as default for high performance/low latency.
    """
    try:
        model = timm.create_model(
            'mobilenetv4_conv_small.e2400_r224_in1k', 
            pretrained=pretrained, 
            num_classes=num_classes
        )
    except Exception as e:
        print(f"Error creating MobileNetV4 from timm: {e}")
        print("Falling back to mobilenetv3_large as placeholder if v4 not found (check timm version)")
        model = timm.create_model(
            'mobilenetv3_large_100', 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
    return model

if __name__ == '__main__':
    model = get_mobilenetv4()
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Created. Total parameters: {total_params/1e6:.2f}M")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
