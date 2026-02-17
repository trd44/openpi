#!/usr/bin/env python3
"""
Simple test to verify the multi-config script works without h5py dependency.
"""
import sys
import os

# Add the project root to the path
sys.path.append('/home/hrilab/Documents/.vlas/cycliclxm-slim/CyclicLxM')

def test_imports():
    """Test that all imports work without h5py."""
    print("Testing imports...")
    
    try:
        from main_hanoi_multi_config import MultiConfigHanoiEnvironment, Args
        print("✅ Imports successful - no h5py dependency!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without running the full environment."""
    print("\nTesting basic functionality...")
    
    try:
        from main_hanoi_multi_config import Args
        
        # Test argument creation
        args = Args(
            episodes=1,
            horizon=100,
            random_block_placement=False,
            random_block_selection=True,
            render_mode="headless"
        )
        print("✅ Args creation successful")
        
        # Test video filename generation
        video_name = args.generate_video_filename(0)
        print(f"✅ Video filename generation: {video_name}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Multi-Config Hanoi (No h5py dependency)")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 All tests passed! The script should work in Docker without h5py.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    sys.exit(0 if success else 1)
