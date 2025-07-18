#!/usr/bin/env python3
"""
Test script for the integrated CrewAI video generation workflow
"""

import os
import sys
from pathlib import Path
from video_crew import VideoGenerationCrew
import logging


def test_workflow():
    """Test the complete workflow without actual media generation"""
    
    print("🧪 Testing CrewAI Video Generation Integration")
    print("=" * 50)
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    fal_key = os.getenv("FAL_AI_API_KEY")
    
    if not openai_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return False
    
    if not fal_key:
        print("❌ FAL_AI_API_KEY not found in environment variables")
        return False
    
    print("✅ Environment variables found")
    
    # Create test output folder
    test_folder = Path("test_output")
    test_folder.mkdir(exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    
    # Test configuration
    config = {
        "OPENAI_API_KEY": openai_key,
        "FAL_AI_API_KEY": fal_key,
        "MAX_CONCURRENT_REQUESTS": 2
    }
    
    # Test prompt
    test_prompt = "Create a 15-second product showcase for a modern smartphone"
    
    try:
        print(f"📝 Test prompt: {test_prompt}")
        
        # Initialize crew
        print("🤖 Initializing CrewAI video generation crew...")
        video_crew = VideoGenerationCrew(config, logger)
        
        # Execute workflow (this will run the first 3 agents)
        print("🚀 Executing CrewAI workflow...")
        results = video_crew.generate_video_content(test_prompt, None, test_folder)
        
        # Check results
        if "summary_result" in results:
            print("✅ Summary generation completed")
        
        if "script_result" in results:
            print("✅ Script generation completed")
        
        if "voice_result" in results:
            print("✅ Voice selection completed")
        
        if "media_result" in results:
            media_status = results["media_result"].get("media_generation_status")
            print(f"✅ Media generation status: {media_status}")
        
        # Save results
        print("💾 Saving results...")
        save_success = video_crew.save_results(results, test_folder)
        
        print("\n📊 Test Results:")
        for filename, success in save_success.items():
            status = "✅" if success else "❌"
            print(f"  {status} {filename}")
        
        print(f"\n📁 Test output folder: {test_folder}")
        print("🎉 Integration test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1) 