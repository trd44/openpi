#!/usr/bin/env python3
"""
Test script for the multi-configuration Hanoi implementation.
This script tests the basic functionality without requiring the OpenPI server.
"""
import os
import sys
import logging
import numpy as np

# Add the project root to the path
sys.path.append('/home/hrilab/Documents/.vlas/cycliclxm-slim/CyclicLxM')

from main_hanoi_multi_config import MultiConfigHanoiEnvironment, Args

def test_environment_setup():
    """Test that the environment can be set up correctly."""
    print("Testing environment setup...")
    
    # Create test arguments
    args = Args(
        episodes=1,
        horizon=100,
        random_block_placement=True,
        random_block_selection=True,
        render_mode="headless"
    )
    
    try:
        # Create environment
        env_manager = MultiConfigHanoiEnvironment(args)
        env_manager.setup()
        print("✅ Environment setup successful")
        
        # Test reset
        obs = env_manager.reset()
        print("✅ Environment reset successful")
        print(f"✅ Generated {len(env_manager.tasks)} tasks from plan")
        
        # Print task details
        for i, task in enumerate(env_manager.tasks):
            print(f"  Task {i+1}: {task['prompt']}")
        
        # Test detector functionality
        if env_manager.detector_ground:
            groundings = env_manager.detector_ground.get_groundings(as_dict=True)
            print(f"✅ Detector working - found {len(groundings)} groundings")
        
        if env_manager.detector_simple:
            status = env_manager.detector_simple.status()
            print(f"✅ Simple detector working - status: {status}")
        
        # Clean up
        env_manager.close()
        print("✅ Environment cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_planner_integration():
    """Test that the planner integration works."""
    print("\nTesting planner integration...")
    
    args = Args(
        episodes=1,
        horizon=100,
        random_block_placement=True,
        random_block_selection=True,
        render_mode="headless"
    )
    
    try:
        env_manager = MultiConfigHanoiEnvironment(args)
        env_manager.setup()
        
        # Test plan generation
        obs = env_manager.reset()
        
        if hasattr(env_manager.recorder, 'plan') and env_manager.recorder.plan:
            print(f"✅ Plan generated successfully: {len(env_manager.recorder.plan)} steps")
            for i, step in enumerate(env_manager.recorder.plan):
                print(f"  Step {i+1}: {step}")
        else:
            print("⚠️  No plan generated, using default tasks")
        
        # Test task sequence generation
        if env_manager.tasks:
            print(f"✅ Task sequence generated: {len(env_manager.tasks)} tasks")
            for i, task in enumerate(env_manager.tasks):
                print(f"  Task {i+1}: {task['prompt']}")
        else:
            print("❌ No tasks generated")
            return False
        
        env_manager.close()
        return True
        
    except Exception as e:
        print(f"❌ Planner integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_configurations():
    """Test with different block configurations."""
    print("\nTesting different configurations...")
    
    configs = [
        {"random_block_placement": True, "random_block_selection": True},
        {"random_block_placement": False, "random_block_selection": True},
        {"random_block_placement": True, "random_block_selection": False},
        {"random_block_placement": False, "random_block_selection": False},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nTesting configuration {i+1}: {config}")
        
        args = Args(
            episodes=1,
            horizon=100,
            render_mode="headless",
            **config
        )
        
        try:
            env_manager = MultiConfigHanoiEnvironment(args)
            env_manager.setup()
            obs = env_manager.reset()
            
            print(f"  ✅ Configuration {i+1} successful - {len(env_manager.tasks)} tasks generated")
            
            env_manager.close()
            
        except Exception as e:
            print(f"  ❌ Configuration {i+1} failed: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Multi-Configuration Hanoi Implementation")
    print("=" * 60)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Planner Integration", test_planner_integration),
        ("Different Configurations", test_different_configurations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The multi-configuration implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
