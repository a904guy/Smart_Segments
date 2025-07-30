"""Smart Segments - AI-powered intelligent segmentation tool for Krita"""

def register_extension():
    """Register the Smart Segments extension with error handling"""
    try:
        from krita import Krita
        from .smart_segments import SmartSegmentsExtension
        
        # Create and register the extension
        extension = SmartSegmentsExtension(Krita.instance())
        Krita.instance().addExtension(extension)
        
        print("Smart Segments extension registered successfully")
        
    except Exception as e:
        print(f"Smart Segments failed to register: {e}")
        import traceback
        traceback.print_exc()
        raise

# Register the extension
register_extension()
